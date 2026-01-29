/**
 * LoRA Megakernel using ThunderKittens.
 *
 * This implements a fully fused kernel for Linear+LoRA:
 *   Y = X @ W.T + (X @ A.T) @ B.T * scale
 *
 * Key optimization: The intermediate S = X @ A.T is kept in SMEM between
 * Phase A (computing S) and Phase B (computing Y = X @ W.T + S @ B.T * scale).
 * This eliminates GMEM traffic for S entirely.
 *
 * Two-phase approach:
 *   Phase A: Compute S = X @ A.T, store in SMEM scratch
 *   Phase B: Compute Y = X @ W.T + S @ B.T * scale, S loaded from SMEM
 *
 * Based on patterns from ThunderKittens mamba2.cu (multi-phase SMEM staging).
 */

#ifdef TORCH_COMPILE
#define TK_COMPILE_MEGAKERNEL
#endif

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// =============================================================================
// Kernel configuration
// =============================================================================

// Tile sizes - optimized for H100 SMEM constraints
// Total SMEM budget: ~227 KB on H100
constexpr int BLOCK_M = 64;   // Rows per block (matches warpgroup)
constexpr int BLOCK_N = 128;  // Output columns per block
constexpr int BLOCK_K = 64;   // K dimension tile
constexpr int MAX_RANK = 256; // Maximum supported LoRA rank

// =============================================================================
// Layout definition
// =============================================================================

template<int _BLOCK_M, int _BLOCK_N, int _BLOCK_K, int _MAX_RANK>
struct lora_megakernel_layout {
    static constexpr int BLOCK_M = _BLOCK_M;
    static constexpr int BLOCK_N = _BLOCK_N;
    static constexpr int BLOCK_K = _BLOCK_K;
    static constexpr int MAX_RANK = _MAX_RANK;

    // Tile types for Phase A: S = X @ A.T
    using X_tile = st_bf<BLOCK_M, BLOCK_K>;     // 64xK for X
    using A_tile = st_bf<BLOCK_K, MAX_RANK>;    // KxR for A (transposed load)

    // Tile types for Phase B: Y = X @ W.T + S @ B.T * scale
    using W_tile = st_bf<BLOCK_K, BLOCK_N>;     // KxN for W (transposed load)
    using S_smem_tile = st_bf<BLOCK_M, MAX_RANK>; // MxR for S (in SMEM!)
    using B_tile = st_bf<MAX_RANK, BLOCK_N>;    // RxN for B (transposed load)
    using O_tile = st_bf<BLOCK_M, BLOCK_N>;     // Output tile

    // Global layouts with TMA descriptors
    struct globals {
        gl<bf16, 1, 1, -1, -1, X_tile> X;  // [1, 1, M, K]
        gl<bf16, 1, 1, -1, -1, W_tile> W;  // [1, 1, K, N]
        gl<bf16, 1, 1, -1, -1, A_tile> A;  // [1, 1, K, R]
        gl<bf16, 1, 1, -1, -1, B_tile> B;  // [1, 1, R, N]
        gl<bf16, 1, 1, -1, -1, O_tile> Y;  // [1, 1, M, N]
        gl<bf16, 1, 1, -1, -1, S_smem_tile> S_out; // [1, 1, M, R] - optional output
        float scale;
        int M, N, K, R;
        bool save_s;  // Whether to save S to GMEM for backward
    };

    // Shared memory input block
    struct input_block {
        X_tile X;      // X tile (reused for both phases)
        A_tile A;      // A tile (Phase A)
        W_tile W;      // W tile (Phase B)
        B_tile B;      // B tile (Phase B)
    };

    // Scratch block - S persists between phases!
    struct scratch_block {
        S_smem_tile S;  // S = X @ A.T, computed in Phase A, used in Phase B
                        // This is the key optimization: S stays in SMEM
    };

    // Register state for consumer
    struct consumer_state {
        rt_fl<16, MAX_RANK> S_acc;   // Accumulator for S (Phase A)
        rt_fl<16, BLOCK_N> Y_acc;    // Accumulator for Y (Phase B)
        int phase;                    // 0 = Phase A, 1 = Phase B
    };

    // Finish block for output
    struct finish_block {
        O_tile Y;  // Output tile
    };
};

// =============================================================================
// Kernel template
// =============================================================================

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int MAX_RANK>
struct lora_megakernel_template {
    using layout = lora_megakernel_layout<BLOCK_M, BLOCK_N, BLOCK_K, MAX_RANK>;

    // Single warpgroup for simplicity (can be extended)
    static constexpr int NUM_CONSUMER_WARPS = 4;  // 1 warpgroup = 4 warps
    static constexpr int NUM_CONSUMER_WARPGROUPS = 1;
    static constexpr int INPUT_PIPE_STAGES = 2;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;
    static constexpr int CONSUMER_BARRIER_ARRIVALS = 1;

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        // Two phases:
        // Phase A: K/BLOCK_K iterations to compute S = X @ A.T
        // Phase B: K/BLOCK_K iterations to compute Y = X @ W.T, plus S @ B computation

        if (args.task_iter == 0) {
            int K = args.globals.K;
            int k_iters = (K + BLOCK_K - 1) / BLOCK_K;
            // Phase A: k_iters iterations
            // Phase B: k_iters iterations + 1 for S @ B
            args.num_iters = 2 * k_iters + 1;
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
                int m_block = blockIdx.x * BLOCK_M;
                int n_block = blockIdx.y * BLOCK_N;

                int K = args.globals.K;
                int R = args.globals.R;
                int k_iters = (K + BLOCK_K - 1) / BLOCK_K;

                bool is_phase_a = args.iter < k_iters;
                bool is_sb_iter = (args.iter == 2 * k_iters);  // Special iteration for S @ B

                if (is_phase_a) {
                    // Phase A: Load X and A
                    int k_offset = args.iter * BLOCK_K;

                    tma::expect(args.inputs_arrived, args.input);
                    tma::load_async(
                        args.input.X, args.globals.X,
                        {0, 0, m_block, k_offset}, args.inputs_arrived
                    );
                    tma::load_async(
                        args.input.A, args.globals.A,
                        {0, 0, k_offset, 0}, args.inputs_arrived
                    );
                    if (laneid() == 0) arrive(args.inputs_arrived, 2);

                } else if (is_sb_iter) {
                    // S @ B iteration: Load B only (S is in SMEM)
                    tma::expect(args.inputs_arrived, args.input);
                    tma::load_async(
                        args.input.B, args.globals.B,
                        {0, 0, 0, n_block}, args.inputs_arrived
                    );
                    if (laneid() == 0) arrive(args.inputs_arrived, 1);

                } else {
                    // Phase B: Load X and W
                    int phase_b_iter = args.iter - k_iters;
                    int k_offset = phase_b_iter * BLOCK_K;

                    tma::expect(args.inputs_arrived, args.input);
                    tma::load_async(
                        args.input.X, args.globals.X,
                        {0, 0, m_block, k_offset}, args.inputs_arrived
                    );
                    tma::load_async(
                        args.input.W, args.globals.W,
                        {0, 0, k_offset, n_block}, args.inputs_arrived
                    );
                    if (laneid() == 0) arrive(args.inputs_arrived, 2);
                }
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            // Initialize accumulators
            zero(args.state.S_acc);
            zero(args.state.Y_acc);
            args.state.phase = 0;  // Start with Phase A
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            int K = args.globals.K;
            int k_iters = (K + BLOCK_K - 1) / BLOCK_K;

            bool is_phase_a = args.iter < k_iters;
            bool is_last_phase_a = (args.iter == k_iters - 1);
            bool is_sb_iter = (args.iter == 2 * k_iters);
            bool is_phase_b = !is_phase_a && !is_sb_iter;

            if (is_phase_a) {
                // ====== PHASE A: Accumulate S = X @ A.T ======
                warpgroup::mma_fence(args.state.S_acc);
                warpgroup::mma_AB(args.state.S_acc, args.input.X, args.input.A);
                warpgroup::mma_async_wait();

                // After last Phase A iteration, store S to SMEM scratch
                if (is_last_phase_a) {
                    // Convert to bf16 and store to scratch SMEM
                    rt_bf<16, MAX_RANK> S_bf16;
                    copy(S_bf16, args.state.S_acc);
                    warpgroup::store(args.scratch.S, S_bf16);
                    warpgroup::sync(0);  // Ensure S is visible

                    // Optionally save S to GMEM for backward
                    if (args.globals.save_s) {
                        int m_block = blockIdx.x * BLOCK_M;
                        if (warpgroup::laneid() == 0) {
                            tma::store_async(
                                args.globals.S_out, args.scratch.S,
                                {0, 0, m_block, 0}
                            );
                        }
                        tma::store_async_read_wait();
                    }
                }

            } else if (is_sb_iter) {
                // ====== S @ B * scale iteration ======
                // Load S from SMEM scratch, compute S @ B, add to Y_acc

                rt_bf<16, MAX_RANK> S_reg;
                warpgroup::load(S_reg, args.scratch.S);
                warpgroup::sync(0);

                rt_fl<16, BLOCK_N> lora_acc;
                zero(lora_acc);
                warpgroup::mma_fence(lora_acc);
                warpgroup::mma_AB(lora_acc, S_reg, args.input.B);
                warpgroup::mma_async_wait();

                // Scale and add to Y_acc
                float scale = args.globals.scale;
                mul(lora_acc, lora_acc, scale);
                add(args.state.Y_acc, args.state.Y_acc, lora_acc);

            } else {
                // ====== PHASE B: Accumulate Y = X @ W.T ======
                warpgroup::mma_fence(args.state.Y_acc);
                warpgroup::mma_AB(args.state.Y_acc, args.input.X, args.input.W);
                warpgroup::mma_async_wait();
            }

            if (laneid() == 0) {
                arrive(args.inputs_finished);
            }
        }

        __device__ static void finish(consumer_finish_args<layout> args) {
            int m_block = blockIdx.x * BLOCK_M;
            int n_block = blockIdx.y * BLOCK_N;

            // Store Y_acc (which now contains Y = X@W.T + S@B.T*scale)
            warpgroup::store(args.finish.Y, args.state.Y_acc);
            warpgroup::sync(0);

            // Write to global memory via TMA
            if (warpgroup::laneid() == 0) {
                tma::store_async(
                    args.globals.Y, args.finish.Y,
                    {0, 0, m_block, n_block}
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

#ifdef TK_COMPILE_MEGAKERNEL
#include "pyutils/torchutils.cuh"
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>

using megakernel_fmt = lora_megakernel_template<BLOCK_M, BLOCK_N, BLOCK_K, MAX_RANK>;
using megakernel_lay = typename megakernel_fmt::layout;

void dispatch_lora_megakernel(
    bf16* d_x,
    bf16* d_w,
    bf16* d_a,
    bf16* d_b,
    bf16* d_y,
    bf16* d_s,
    float scale,
    int M, int N, int K, int R,
    bool save_s
) {
    // Setup global layouts
    typename megakernel_lay::globals G;

    G.X = kittens::make_gl<decltype(G.X)>((uint64_t)d_x, 1, 1, M, K);
    G.W = kittens::make_gl<decltype(G.W)>((uint64_t)d_w, 1, 1, K, N);
    G.A = kittens::make_gl<decltype(G.A)>((uint64_t)d_a, 1, 1, K, R);
    G.B = kittens::make_gl<decltype(G.B)>((uint64_t)d_b, 1, 1, R, N);
    G.Y = kittens::make_gl<decltype(G.Y)>((uint64_t)d_y, 1, 1, M, N);

    if (save_s && d_s != nullptr) {
        G.S_out = kittens::make_gl<decltype(G.S_out)>((uint64_t)d_s, 1, 1, M, R);
    }

    G.scale = scale;
    G.M = M;
    G.N = N;
    G.K = K;
    G.R = R;
    G.save_s = save_s && d_s != nullptr;

    // Launch configuration
    unsigned long mem_size = kittens::prototype::detail::MAX_SHARED_MEMORY_v<megakernel_fmt>;
    cudaFuncSetAttribute(
        prototype::lcf::kernel<megakernel_fmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    // Grid: (M/BLOCK_M, N/BLOCK_N)
    int m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    int n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(m_blocks, n_blocks);
    dim3 block(prototype::detail::NUM_THREADS_v<megakernel_fmt>);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    prototype::lcf::kernel<megakernel_fmt><<<grid, block, mem_size, stream>>>(G);
}

std::tuple<at::Tensor, at::Tensor> lora_megakernel_forward(
    const at::Tensor x,    // (M, K)
    const at::Tensor W,    // (N, K)
    const at::Tensor A,    // (R, K)
    const at::Tensor B,    // (N, R)
    float scale,
    bool save_s
) {
    CHECK_INPUT(x);
    CHECK_INPUT(W);
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = x.size(0);
    int K = x.size(1);
    int N = W.size(0);
    int R = A.size(0);

    TORCH_CHECK(W.size(1) == K, "W has incompatible K dimension");
    TORCH_CHECK(A.size(1) == K, "A has incompatible K dimension");
    TORCH_CHECK(B.size(0) == N, "B has incompatible N dimension");
    TORCH_CHECK(B.size(1) == R, "B has incompatible R dimension");
    TORCH_CHECK(R <= MAX_RANK, "LoRA rank exceeds maximum supported rank");

    // Allocate outputs
    at::Tensor Y = at::empty({M, N}, x.options());
    at::Tensor S = save_s ? at::empty({M, R}, x.options()) : at::Tensor();

    // Get data pointers
    bf16* d_x = reinterpret_cast<bf16*>(x.data_ptr<c10::BFloat16>());
    bf16* d_w = reinterpret_cast<bf16*>(W.data_ptr<c10::BFloat16>());
    bf16* d_a = reinterpret_cast<bf16*>(A.data_ptr<c10::BFloat16>());
    bf16* d_b = reinterpret_cast<bf16*>(B.data_ptr<c10::BFloat16>());
    bf16* d_y = reinterpret_cast<bf16*>(Y.data_ptr<c10::BFloat16>());
    bf16* d_s = save_s ? reinterpret_cast<bf16*>(S.data_ptr<c10::BFloat16>()) : nullptr;

    dispatch_lora_megakernel(d_x, d_w, d_a, d_b, d_y, d_s, scale, M, N, K, R, save_s);

    CHECK_CUDA_ERROR(cudaGetLastError());

    return std::make_tuple(Y, S);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &lora_megakernel_forward,
        "LoRA megakernel forward: Y = X @ W.T + (X @ A.T) @ B.T * scale\n"
        "Returns (Y, S) where S is the intermediate if save_s=True"
    );
}
#endif
