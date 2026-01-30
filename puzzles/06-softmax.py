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
