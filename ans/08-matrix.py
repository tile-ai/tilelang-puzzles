"""
Puzzle 08: Matrix Computation
==============
We now start to solve one of the most basic workloads of deep learning in TileLang.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
This chapter contains two puzzles: (1) matrix-vector multiplication (GEMV) and
(2) matrix-matrix multiplication (GEMM). GEMV can be implemented by extending our previous
"reduce sum" example.

NOTE: Modern AI workloads usually use float16 as the default data type. Hence, we will use float16
as the input/output dtype in this puzzle, with a seperate high-precision accumulator dtype like
float32.

08-1: Matrix-Vector Multiplication.

Inputs:
    A: [M, K]  # input matrix
    B: [K,]  # input vector
    N: int   # size of the tensor. 1 <= N <= 8192
    K: int   # size of the tensor. 1 <= K <= 8192
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    C: [M,]  # output tensor

Definition:
    for i in range(M):
        C[i] = 0
        for k in range(K):
            C[i] += A[i, k] * B[k]
"""


def ref_gemv(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 1
    assert A.shape[1] == B.shape[0]  # K
    assert A.dtype == B.dtype == torch.float16
    return torch.matmul(input=A, other=B)


@tilelang.jit
def tl_gemv(A, B, BLOCK_M: int, BLOCK_K: int):
    M, K = T.const("M, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K,), dtype)
    C = T.empty((M,), dtype)

    # TODO: Implement this function
    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=128) as pid_m:
        A_local = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
        B_local = T.alloc_fragment((BLOCK_K,), dtype)
        C_local = T.alloc_fragment((BLOCK_M,), accum_dtype)

        AB_temp = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)

        T.clear(C_local)
        for k in T.Serial(K // BLOCK_K):
            T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_local)
            T.copy(B[k * BLOCK_K,], B_local)

            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                AB_temp[i, j] = A_local[i, j].astype(accum_dtype) * B_local[j].astype(accum_dtype)

            T.reduce_sum(AB_temp, C_local, dim=1, clear=False)

        T.copy(C_local, C[pid_m * BLOCK_M,])

    return C


def run_gemv():
    print("\n=== Matrix-Vector Multiplication ===\n")

    M = 4096
    K = 4096
    BLOCK_M = 128
    BLOCK_K = 32

    test_puzzle(tl_gemv, ref_gemv, {"M": M, "K": K, "BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K})
    bench_puzzle(
        tl_gemv,
        ref_gemv,
        {"M": M, "K": K, "BLOCK_M": BLOCK_M, "BLOCK_K": BLOCK_K},
        bench_torch=True,
    )


"""
From GEMV to GEMM, the actual complexity of the problem grows exponentially. There are many
optimizations you need to know if you want to implement a high-performance matmul kernel matched
with cuBLAS, like pipelining, swizzling, tiling, etc. But with TileLang, we can focus on the
dataflow and tiling computation.

In modern GPU like NVIDIA Hopper architecture, there are specialized units for matrix
multiplication called Tensor Cores. They can perform operations like 16x16x16 FP16 tensor core
operation, which is called a MMA instruction. In previous examples, most of our computations are
performed on CUDA Cores, which are efficient for scalar/vector operations. However, Tensor Cores
are optimized for matrix operations and can achieve much higher throughput for large matrices.

TileLang wraps these complex instructions and memory loading patterns into a simple `T.gemm`
operator that can be used to generate high-performance matrix multiplication kernels. `T.gemm`
takes two Buffers as input and one Buffer as output, just like other TileOp we have seen before.
The rest thing is just to tile the whole matrix.

08-2: Matmul (Matrix-Matrix Multiplication)

Inputs:
    A: [M, K]  # input tensor
    B: [K, N]  # input tensor
    N: int   # size of the tensor. 1 <= N <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    C: [M, N]  # output tensor

Definition:
    for i in range(M):
        for j in range(N):
            C[i, j] = 0
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
"""


def ref_matmul(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 2
    assert A.shape[1] == B.shape[0]  # K
    assert A.dtype == B.dtype == torch.float16
    return torch.matmul(input=A, other=B)


@tilelang.jit
def tl_matmul_naive(A, B, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    # TODO: Implement this function
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (
        bx,
        by,
    ):
        A_local = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
        B_local = T.alloc_fragment((BLOCK_K, BLOCK_N), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.clear(C_local)
        for k in T.Serial(K // BLOCK_K):
            T.copy(A[by * BLOCK_M, k * BLOCK_K], A_local)
            T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_local)
            T.gemm(A_local, B_local, C_local)

        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    return C


def run_matmul_naive():
    print("\n=== Matrix Multiplication Naive ===\n")

    M = 4096
    N = 4096
    K = 4096
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    test_puzzle(
        tl_matmul_naive,
        ref_matmul,
        {
            "M": M,
            "N": N,
            "K": K,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
        },
    )
    bench_puzzle(
        tl_matmul_naive,
        ref_matmul,
        {
            "M": M,
            "N": N,
            "K": K,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
        },
        bench_torch=True,
    )


"""
Previous implementation works but the performance is not optimal. Here we introduce two
optimizations here, with just a few lines of code changes.

1. Shared Memory Optimization. We use fragment as the intermediate buffer in all previous puzzles
without detailed explanation since we want to keep the tutorial simple. However, recall that the
essential of fragment is the unified memory abstraction of registers in all threads. If we put
A, B, C tiles all in registers, the registers will be exhausted quickly and cause register
spilling. Therefore, we need to use shared memory to store the tiles of A and B. `T.gemm` will
efficiently help us load data from shared memory, so we can directly use `T.alloc_shared` to
allocate shared memory for A and B tiles.

2. Software Pipeline. Starting from the NVIDIA Ampere architecture, software pipeline is an
important optimization technique to overlap computation and memory access. In our case, we can use
software pipeline to overlap the loading of A and B tiles with the computation of the GEMM
operation. This is achieved by using `T.Pipeline` to replace `T.Serial` and specifying a proper
stage number, like num_stage=3.

After modifying the code, we can take a look at the generated CUDA code and compare the performance
improvement.
"""


@tilelang.jit
def tl_matmul_opt(A, B, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    # TODO: Implement this function
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (
        bx,
        by,
    ):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(K // BLOCK_K, num_stages=3):
            T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
            T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    return C


def run_matmul_opt():
    print("\n=== Matrix Multiplication ===\n")

    M = 4096
    N = 4096
    K = 4096
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    args_dict = {
        "M": M,
        "N": N,
        "K": K,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
    }

    print("Naive Matmul Implementation: ")
    naive_matmul_kernel = tl_matmul_naive.compile(**args_dict)
    naive_matmul_kernel.print_source_code()

    print("OPT Matmul Implementation: ")
    opt_matmul_kernel = tl_matmul_opt.compile(**args_dict)
    opt_matmul_kernel.print_source_code()

    bench_puzzle(tl_matmul_naive, ref_matmul, args_dict, bench_torch=True)
    bench_puzzle(tl_matmul_opt, ref_matmul, args_dict, bench_torch=True)


if __name__ == "__main__":
    run_gemv()
    run_matmul_naive()
    run_matmul_opt()
