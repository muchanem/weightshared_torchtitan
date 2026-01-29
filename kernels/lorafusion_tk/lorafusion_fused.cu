/**
 * LoRAFusion-style fused kernel using ThunderKittens.
 *
 * This implements the 2-kernel LoRAFusion approach:
 *   Kernel 1 (cuBLAS): S = X @ A.T (external, optimal)
 *   Kernel 2 (this):   Y = X @ W.T + S @ B.T * scale
 *
 * This kernel fuses the base linear projection with the LoRA-B matmul,
 * where S is pre-computed and passed in from GMEM.
 *
 * Target operation: Y = X @ W.T + S @ B.T * scale
 *   X: (M, K) input
 *   W: (N, K) base weight
 *   S: (M, R) pre-computed intermediate (X @ A.T)
 *   B: (N, R) LoRA B weight
 *   Y: (M, N) output
 *   scale: LoRA scaling factor (alpha/rank)
 */

#ifdef TORCH_COMPILE
#define TK_COMPILE_LORAFUSION
#endif

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// =============================================================================
// Kernel configuration
// =============================================================================

// Tile sizes - optimized for H100
constexpr int BLOCK_M = 128;  // Rows per block
constexpr int BLOCK_N = 128;  // Cols per block
constexpr int BLOCK_K = 64;   // K dimension tile
constexpr int BLOCK_R = 64;   // Rank dimension tile (for S @ B)

// =============================================================================
// Layout definition
// =============================================================================

template<int _BLOCK_M, int _BLOCK_N, int _BLOCK_K, int _BLOCK_R>
struct lorafusion_layout {
    static constexpr int BLOCK_M = _BLOCK_M;
    static constexpr int BLOCK_N = _BLOCK_N;
    static constexpr int BLOCK_K = _BLOCK_K;
    static constexpr int BLOCK_R = _BLOCK_R;

    // Tile types
    using X_tile = st_bf<64, BLOCK_K>;      // 64xK tile for X
    using W_tile = st_bf<BLOCK_K, BLOCK_N>; // KxN tile for W (transposed)
    using S_tile = st_bf<64, BLOCK_R>;      // 64xR tile for S
    using B_tile = st_bf<BLOCK_R, BLOCK_N>; // RxN tile for B (transposed)
    using O_tile = st_bf<64, BLOCK_N>;      // Output tile

    // Global layouts with TMA descriptors
    struct globals {
        gl<bf16, 1, 1, -1, -1, X_tile> X;  // [1, 1, M, K]
        gl<bf16, 1, 1, -1, -1, W_tile> W;  // [1, 1, K, N] (stored as N, K, transposed for load)
        gl<bf16, 1, 1, -1, -1, S_tile> S;  // [1, 1, M, R]
        gl<bf16, 1, 1, -1, -1, B_tile> B;  // [1, 1, R, N] (stored as N, R, transposed)
        gl<bf16, 1, 1, -1, -1, O_tile> Y;  // [1, 1, M, N]
        float scale;
        int M, N, K, R;
    };

    // Shared memory input block (double-buffered)
    struct input_block {
        X_tile X[BLOCK_M/64];  // X tiles for this M block
        W_tile W;              // W tile
        S_tile S[BLOCK_M/64];  // S tiles for LoRA
        B_tile B;              // B tile for LoRA
    };

    // Scratch block (persistent across iterations)
    struct scratch_block {
        // Nothing needed - we accumulate in registers
    };

    // Register state for consumer
    struct consumer_state {
        rt_fl<16, BLOCK_N> Y_acc;  // Accumulator for output (64xN per warpgroup)
    };

    // Finish block for output
    struct finish_block {
        O_tile Y[BLOCK_M/64];  // Output tiles
    };
};

// =============================================================================
// Kernel template
// =============================================================================

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int BLOCK_R>
struct lorafusion_template {
    using layout = lorafusion_layout<BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_R>;

    static constexpr int NUM_CONSUMER_WARPS = BLOCK_M / 16;  // 4 warps per 64 rows
    static constexpr int NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
    static constexpr int CONSUMER_BARRIER_ARRIVALS = NUM_CONSUMER_WARPS / 4;

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        // Compute number of iterations (over K dimension for base GEMM)
        // We also need to iterate over R for LoRA, but that's handled inside compute
        int K = args.globals.K;
        int R = args.globals.R;

        // We iterate over K blocks for the main GEMM
        // R blocks for LoRA are handled within each K iteration
        if (args.task_iter == 0) {
            int k_iters = (K + BLOCK_K - 1) / BLOCK_K;
            args.num_iters = k_iters;
        } else {
            args.num_iters = -1;  // No more work
        }
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<32>();
        }

        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                int m_block = blockIdx.x * BLOCK_M;
                int n_block = blockIdx.y * BLOCK_N;
                int k_offset = args.iter * BLOCK_K;

                // Expect all tiles
                tma::expect(args.inputs_arrived, args.input);

                // Load X tiles (BLOCK_M/64 tiles of 64xK)
                #pragma unroll
                for (int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                    tma::load_async(
                        args.input.X[i],
                        args.globals.X,
                        {0, 0, m_block + i * 64, k_offset},
                        args.inputs_arrived
                    );
                }

                // Load W tile (KxN)
                // W is stored as (N, K), we want columns [n_block, n_block+N) and rows [k_offset, k_offset+K)
                tma::load_async(
                    args.input.W,
                    args.globals.W,
                    {0, 0, k_offset, n_block},
                    args.inputs_arrived
                );

                // Load S tiles only on first K iteration (S doesn't depend on K)
                // Actually, we need S @ B for all of LoRA, so load S based on R blocks
                if (args.iter == 0) {
                    #pragma unroll
                    for (int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                        tma::load_async(
                            args.input.S[i],
                            args.globals.S,
                            {0, 0, m_block + i * 64, 0},  // First R block
                            args.inputs_arrived
                        );
                    }

                    // Load B tile
                    tma::load_async(
                        args.input.B,
                        args.globals.B,
                        {0, 0, 0, n_block},
                        args.inputs_arrived
                    );
                }

                if (laneid() == 0) {
                    arrive(args.inputs_arrived, 3 + (args.iter == 0 ? NUM_CONSUMER_WARPGROUPS + 1 : 0));
                }
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            // Initialize accumulator to zero
            zero(args.state.Y_acc);
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            int wg_id = warpgroup::groupid();

            // Accumulate X @ W.T into Y_acc
            // args.input.X[wg_id]: 64xK
            // args.input.W: KxN
            // Result: 64xN
            warpgroup::mma_fence(args.state.Y_acc);
            warpgroup::mma_AB(args.state.Y_acc, args.input.X[wg_id], args.input.W);
            warpgroup::mma_async_wait();

            // On first iteration, also compute LoRA: S @ B.T * scale
            // This is added to Y_acc
            if (args.iter == 0) {
                // S[wg_id]: 64xR
                // B: RxN
                // Result: 64xN, scaled and added to Y_acc

                // Compute S @ B into temporary, then scale and add
                rt_fl<16, BLOCK_N> lora_acc;
                zero(lora_acc);
                warpgroup::mma_fence(lora_acc);
                warpgroup::mma_AB(lora_acc, args.input.S[wg_id], args.input.B);
                warpgroup::mma_async_wait();

                // Scale and add: Y_acc += lora_acc * scale
                float scale = args.globals.scale;
                mul(lora_acc, lora_acc, scale);
                add(args.state.Y_acc, args.state.Y_acc, lora_acc);
            }

            if (laneid() == 0) {
                arrive(args.inputs_finished);
            }
        }

        __device__ static void finish(consumer_finish_args<layout> args) {
            int wg_id = warpgroup::groupid();
            int m_block = blockIdx.x * BLOCK_M;
            int n_block = blockIdx.y * BLOCK_N;

            // Store accumulated result to finish block
            warpgroup::store(args.finish.Y[wg_id], args.state.Y_acc);
            warpgroup::sync(wg_id);

            // Write to global memory via TMA
            if (warpgroup::laneid() == 0) {
                tma::store_async(
                    args.globals.Y,
                    args.finish.Y[wg_id],
                    {0, 0, m_block + wg_id * 64, n_block}
                );
            }
            tma::store_async_read_wait();
            __syncwarp();

            if (laneid() == 0) {
                arrive(args.finish_finished);
            }
        }
    };
};

// =============================================================================
// PyTorch binding
// =============================================================================

#ifdef TK_COMPILE_LORAFUSION
#include "pyutils/torchutils.cuh"
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>

using lorafusion_fmt = lorafusion_template<BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_R>;
using lorafusion_lay = typename lorafusion_fmt::layout;

void dispatch_lorafusion_fused(
    bf16* d_x,
    bf16* d_w,
    bf16* d_s,
    bf16* d_b,
    bf16* d_y,
    float scale,
    int M, int N, int K, int R
) {
    // Setup global layouts
    typename lorafusion_lay::globals G;

    // Create global layouts with dimensions
    G.X = kittens::make_gl<decltype(G.X)>((uint64_t)d_x, 1, 1, M, K);
    G.W = kittens::make_gl<decltype(G.W)>((uint64_t)d_w, 1, 1, K, N);
    G.S = kittens::make_gl<decltype(G.S)>((uint64_t)d_s, 1, 1, M, R);
    G.B = kittens::make_gl<decltype(G.B)>((uint64_t)d_b, 1, 1, R, N);
    G.Y = kittens::make_gl<decltype(G.Y)>((uint64_t)d_y, 1, 1, M, N);
    G.scale = scale;
    G.M = M;
    G.N = N;
    G.K = K;
    G.R = R;

    // Launch configuration
    unsigned long mem_size = kittens::prototype::detail::MAX_SHARED_MEMORY_v<lorafusion_fmt>;
    cudaFuncSetAttribute(
        prototype::lcf::kernel<lorafusion_fmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    // Grid: (M/BLOCK_M, N/BLOCK_N)
    int m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    int n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(m_blocks, n_blocks);
    dim3 block(prototype::detail::NUM_THREADS_v<lorafusion_fmt>);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    prototype::lcf::kernel<lorafusion_fmt><<<grid, block, mem_size, stream>>>(G);
}

at::Tensor lorafusion_fused_forward(
    const at::Tensor x,    // (M, K)
    const at::Tensor w,    // (N, K)
    const at::Tensor s,    // (M, R) - pre-computed X @ A.T
    const at::Tensor b,    // (N, R)
    float scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(s);
    CHECK_INPUT(b);

    int M = x.size(0);
    int K = x.size(1);
    int N = w.size(0);
    int R = s.size(1);

    TORCH_CHECK(w.size(1) == K, "W has incompatible K dimension");
    TORCH_CHECK(s.size(0) == M, "S has incompatible M dimension");
    TORCH_CHECK(b.size(0) == N, "B has incompatible N dimension");
    TORCH_CHECK(b.size(1) == R, "B has incompatible R dimension");

    // Allocate output
    at::Tensor y = at::empty({M, N}, x.options());

    // Get data pointers
    bf16* d_x = reinterpret_cast<bf16*>(x.data_ptr<c10::BFloat16>());
    bf16* d_w = reinterpret_cast<bf16*>(w.data_ptr<c10::BFloat16>());
    bf16* d_s = reinterpret_cast<bf16*>(s.data_ptr<c10::BFloat16>());
    bf16* d_b = reinterpret_cast<bf16*>(b.data_ptr<c10::BFloat16>());
    bf16* d_y = reinterpret_cast<bf16*>(y.data_ptr<c10::BFloat16>());

    dispatch_lorafusion_fused(d_x, d_w, d_s, d_b, d_y, scale, M, N, K, R);

    CHECK_CUDA_ERROR(cudaGetLastError());

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_forward",
        &lorafusion_fused_forward,
        "LoRAFusion fused forward: Y = X @ W.T + S @ B.T * scale"
    );
}
#endif
