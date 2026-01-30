"""
Puzzle 06: Softmax
==============
Softmax is the first fundermental NN operator we learn in this tutorial.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

r"""
Softmax operator goes a little beyond the reduce sum. We also need to use serial loop to
accumulate the summation. And we need to perform an element-wise exp operation on each element
at the same time.

Note that softmax needs to be computed in numerically stable form as in Python. To achieve this,
we need to subtract the maximum value of each row from all elements in that row
before applying the exponential function.

HINT:
1. Use `T.fill` to set the initial value of the buffer. `T.clear` sets all elements to zero by
default, which may not be what you want.

3.We recommend not using `T.exp` but instead using `T.exp2`. You need the identity

.. math::
    \exp(x) = 2^{\log_2(e) x}

The constant log2_e is provided.

BONUS: Use "Online Softmax" algorithm to implement optimized softmax. This is also a core idea of
FlashAttention algorithm. Through this, we can implement softmax with only two passes / loops.

06-1: Softmax.

Inputs:
    A: [N, M]  # input tensor
    N: int   # size of the tensor. 1 <= N <= 4096
    M: int   # size of the tensor. 1 <= M <= 16384
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    B: [N, M]  # output tensor

Intermediates:
    MAX: dtype  # max value of each row
    SUM: dtype  # summation of each row

Definition:
    for i in range(N):
        S = 0
        MAX = -inf
        for j in range(M):
            MAX = max(A[i, j], MAX)
        for j in range(M):
            B[i, j] = exp(A[i, j] - MAX)
            SUM += B[i, j]
        for j in range(M):
            B[i, j] /= SUM
"""


def ref_softmax(A: torch.Tensor):
    assert len(A.shape) == 2
    assert A.dtype == torch.float32
    return torch.softmax(A, dim=1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_softmax(A, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    # TODO: Implement this function
    with T.Kernel(N // BLOCK_N, threads=256) as pid_n:
        A_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)

        # These three buffers are updated every BLOCK_M iteration,
        # so we name them with a cur_ prefix.
        cur_exp_A = T.alloc_fragment([BLOCK_N, BLOCK_M], dtype)
        cur_max_A = T.alloc_fragment([BLOCK_N], dtype)
        cur_sum_exp_A = T.alloc_fragment([BLOCK_N], dtype)

        # LSE is short for log-sum-exp, and it's not per block.
        # It's the output of our first serial loop.
        lse = T.alloc_fragment([BLOCK_N], dtype)

        T.fill(lse, -T.infinity(dtype))

        # The first loop use an online algorithm to compute LSE.
        for m_blk_id in T.Serial(M // BLOCK_M):
            T.copy(A[pid_n * BLOCK_N, m_blk_id * BLOCK_M], A_local)
            T.reduce_max(A_local, cur_max_A, dim=1, clear=True)

            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                cur_exp_A[i, j] = T.exp2(A_local[i, j] * log2_e - cur_max_A[i] * log2_e)

            T.reduce_sum(cur_exp_A, cur_sum_exp_A, dim=1, clear=True)

            for i in T.Parallel(BLOCK_N):
                lse[i] = cur_max_A[i] * log2_e + T.log2(
                    T.exp2(lse[i] - cur_max_A[i] * log2_e) + cur_sum_exp_A[i]
                )

        # The second loop use LSE to get the final output.
        for m_blk_id in T.Serial(M // BLOCK_M):
            T.copy(A[pid_n * BLOCK_N, m_blk_id * BLOCK_M], A_local)

            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                B_local[i, j] = T.exp2(A_local[i, j] * log2_e - lse[i])

            T.copy(B_local, B[pid_n * BLOCK_N, m_blk_id * BLOCK_M])

    return B


def run_softmax():
    print("\n=== Softmax ===\n")
    N = 4096
    M = 16384
    BLOCK_N = 16
    BLOCK_M = 256
    test_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    bench_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_softmax()
