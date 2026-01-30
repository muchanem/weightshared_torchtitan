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
 * Design: Simplified from flux_gate.cu pattern
 * - Stream through K for X @ W.T
 * - On first K iteration, also load and compute S @ B.T (full rank at once)
 * - Add scaled LoRA contribution to output
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

// Tile sizes optimized for H100
// BLOCK_M = 64 * NUM_CONSUMER_WARPGROUPS
constexpr int BLOCK_M = 128;  // Rows per block (2 warpgroups x 64)
constexpr int BLOCK_N = 128;  // Cols per block
constexpr int BLOCK_K = 64;   // K dimension tile
constexpr int BLOCK_R = 64;   // Rank dimension for S @ B (tiles over R)

// =============================================================================
// Layout definition (following flux_gate.cu pattern)
// =============================================================================

template<int _BLOCK_M, int _BLOCK_N, int _BLOCK_K>
struct lorafusion_layout {
    // Tile types (64 rows per warpgroup for Hopper WGMMA)
    // For mma_ABt(Y, X, W): computes Y = X @ W.T
    // X: (64, K), W stored as (N, K) for transposed access
    using x_tile = st_bf<64, _BLOCK_K>;       // X tile: 64 x K
    using w_tile = st_bf<_BLOCK_N, _BLOCK_K>; // W tile: N x K (row-major, for ABt)
    using s_tile = st_bf<64, BLOCK_R>;        // S tile: 64 x R
    using b_tile = st_bf<_BLOCK_N, BLOCK_R>;  // B tile: N x R (row-major, for ABt)
    using o_tile = st_bf<64, _BLOCK_N>;       // Output tile: 64 x N

    // Type aliases for globals
    using x_global = gl<bf16, 1, 1, -1, -1, x_tile>;
    using w_global = gl<bf16, 1, 1, -1, -1, w_tile>;
    using s_global = gl<bf16, 1, 1, -1, -1, s_tile>;
    using b_global = gl<bf16, 1, 1, -1, -1, b_tile>;
    using o_global = gl<bf16, 1, 1, -1, -1, o_tile>;

    struct globals {
        x_global X;
        w_global W;
        s_global S;
        b_global B;
        o_global Y;
        float scale;
    };

    struct input_block {
        x_tile x[_BLOCK_M/64];   // X tiles for each warpgroup
        w_tile w;                 // W tile (shared across warpgroups)
        s_tile s[_BLOCK_M/64];   // S tiles for LoRA
        b_tile b;                 // B tile for LoRA (shared)
    };

    struct scratch_block {
        // Empty - no persistent scratch needed
    };

    struct consumer_state {
        rt_fl<16, _BLOCK_N> y_acc;  // Accumulator for output
    };

    struct finish_block {
        o_tile o[_BLOCK_M/64];  // Output tiles for store
    };
};

// =============================================================================
// Kernel template
// =============================================================================

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct lorafusion_template {
    using layout = lorafusion_layout<BLOCK_M, BLOCK_N, BLOCK_K>;

    static constexpr int NUM_CONSUMER_WARPS = BLOCK_M / 16;  // 4 warps per 64-row chunk
    static constexpr int NUM_CONSUMER_WARPGROUPS = BLOCK_M / 64;
    static constexpr int INPUT_PIPE_STAGES = 3;

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if (args.task_iter == 0) {
            // Number of K iterations
            int k_iters = args.globals.W.rows() / BLOCK_K;
            args.num_iters = k_iters;
        } else {
            args.num_iters = -1;
        }
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<32>();
        }

        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                // Block indices (in tile coordinates)
                int m_tile = blockIdx.x * (BLOCK_M / 64);  // M tile index
                int n_tile = blockIdx.y;                    // N tile index
                int k_tile = args.iter;                     // K iteration (tile index)

                // Calculate expected bytes based on what we're loading
                // iter 0: X + W + S + B tiles
                // iter > 0: X + W tiles only
                constexpr size_t x_bytes = sizeof(typename layout::x_tile) * NUM_CONSUMER_WARPGROUPS;
                constexpr size_t w_bytes = sizeof(typename layout::w_tile);
                constexpr size_t s_bytes = sizeof(typename layout::s_tile) * NUM_CONSUMER_WARPGROUPS;
                constexpr size_t b_bytes = sizeof(typename layout::b_tile);

                if (args.iter == 0) {
                    tma::expect_bytes(args.inputs_arrived, x_bytes + w_bytes + s_bytes + b_bytes);
                } else {
                    tma::expect_bytes(args.inputs_arrived, x_bytes + w_bytes);
                }

                // Load X tiles: X is (M, K), x_tile is (64, BLOCK_K)
                // TMA coords: {batch, depth, row_tile, col_tile}
                #pragma unroll
                for (int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                    tma::load_async(
                        args.input.x[i], args.globals.X,
                        {0, 0, m_tile + i, k_tile},
                        args.inputs_arrived
                    );
                }

                // Load W tile: W is (N, K), w_tile is (BLOCK_N, BLOCK_K)
                // We want the n_tile-th N block, k_tile-th K block
                tma::load_async(
                    args.input.w, args.globals.W,
                    {0, 0, n_tile, k_tile},
                    args.inputs_arrived
                );

                // On first iteration, also load S and B for LoRA
                if (args.iter == 0) {
                    // Load S tiles: S is (M, R), s_tile is (64, BLOCK_R)
                    #pragma unroll
                    for (int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                        tma::load_async(
                            args.input.s[i], args.globals.S,
                            {0, 0, m_tile + i, 0},
                            args.inputs_arrived
                        );
                    }

                    // Load B tile: B is (N, R), b_tile is (BLOCK_N, BLOCK_R)
                    tma::load_async(
                        args.input.b, args.globals.B,
                        {0, 0, n_tile, 0},
                        args.inputs_arrived
                    );
                }

                if (laneid() == 0) arrive(args.inputs_arrived, 3);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<NUM_CONSUMER_WARPGROUPS == 3 ? 152 : 232>();
            warp::zero(args.state.y_acc);
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            int wg_id = warpgroup::groupid();

            // Accumulate X @ W.T
            warpgroup::mma_ABt(args.state.y_acc, args.input.x[wg_id], args.input.w);
            warpgroup::mma_async_wait();

            // On first iteration, also compute S @ B.T * scale
            if (args.iter == 0) {
                rt_fl<16, BLOCK_N> lora_acc;
                warp::zero(lora_acc);
                warpgroup::mma_ABt(lora_acc, args.input.s[wg_id], args.input.b);
                warpgroup::mma_async_wait();

                // Scale and add to y_acc
                float scale = args.globals.scale;
                warp::mul(lora_acc, lora_acc, scale);
                warp::add(args.state.y_acc, args.state.y_acc, lora_acc);
            }

            if (laneid() == 0) arrive(args.inputs_finished);
        }

        __device__ static void finish(consumer_finish_args<layout> args) {
            int wg_id = warpgroup::groupid();
            // Tile coordinates for output
            int m_tile = blockIdx.x * NUM_CONSUMER_WARPGROUPS + wg_id;
            int n_tile = blockIdx.y;

            // Store to finish block
            warpgroup::store(args.finish.o[wg_id], args.state.y_acc);
            warpgroup::sync(wg_id);

            // TMA store to global memory
            if (warpgroup::laneid() == 0) {
                tma::store_async(
                    args.globals.Y, args.finish.o[wg_id],
                    {0, 0, m_tile, n_tile}
                );
            }
            tma::store_async_read_wait();
            __syncwarp();

            if (laneid() == 0) arrive(args.finish_finished);
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

using lorafusion_fmt = lorafusion_template<BLOCK_M, BLOCK_N, BLOCK_K>;
using lorafusion_lay = typename lorafusion_fmt::layout;

// Type aliases for cleaner code
using x_global = typename lorafusion_lay::x_global;
using w_global = typename lorafusion_lay::w_global;
using s_global = typename lorafusion_lay::s_global;
using b_global = typename lorafusion_lay::b_global;
using o_global = typename lorafusion_lay::o_global;
using globals = typename lorafusion_lay::globals;

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
    TORCH_CHECK(M % BLOCK_M == 0, "M must be divisible by BLOCK_M");
    TORCH_CHECK(N % BLOCK_N == 0, "N must be divisible by BLOCK_N");
    TORCH_CHECK(K % BLOCK_K == 0, "K must be divisible by BLOCK_K");
    TORCH_CHECK(R == BLOCK_R, "R must equal BLOCK_R (64) for now");

    // Allocate output
    at::Tensor y = at::empty({M, N}, x.options());

    // Create global layout objects
    bf16* d_x = reinterpret_cast<bf16*>(x.data_ptr<c10::BFloat16>());
    bf16* d_w = reinterpret_cast<bf16*>(w.data_ptr<c10::BFloat16>());
    bf16* d_s = reinterpret_cast<bf16*>(s.data_ptr<c10::BFloat16>());
    bf16* d_b = reinterpret_cast<bf16*>(b.data_ptr<c10::BFloat16>());
    bf16* d_y = reinterpret_cast<bf16*>(y.data_ptr<c10::BFloat16>());

    x_global Xg{d_x, nullptr, nullptr, M, K};
    w_global Wg{d_w, nullptr, nullptr, K, N};
    s_global Sg{d_s, nullptr, nullptr, M, R};
    b_global Bg{d_b, nullptr, nullptr, R, N};
    o_global Yg{d_y, nullptr, nullptr, M, N};

    // Aggregate initialize globals
    globals G{Xg, Wg, Sg, Bg, Yg, scale};

    // Launch configuration
    unsigned long mem_size = kittens::prototype::detail::MAX_SHARED_MEMORY_v<lorafusion_fmt>;
    cudaFuncSetAttribute(
        prototype::lcf::kernel<lorafusion_fmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    int m_blocks = M / BLOCK_M;
    int n_blocks = N / BLOCK_N;
    dim3 grid(m_blocks, n_blocks);
    dim3 block(prototype::detail::NUM_THREADS_v<lorafusion_fmt>);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    prototype::lcf::kernel<lorafusion_fmt><<<grid, block, mem_size, stream>>>(G);

    CHECK_CUDA_ERROR(cudaGetLastError());

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_forward",
        &lorafusion_fused_forward,
        "LoRAFusion fused forward: Y = X @ W.T + S @ B.T * scale\n"
        "Args:\n"
        "  x: Input tensor (M, K)\n"
        "  w: Base weight (N, K)\n"
        "  s: Pre-computed S = X @ A.T (M, R)\n"
        "  b: LoRA B weight (N, R)\n"
        "  scale: LoRA scaling factor"
    );
}
#endif
