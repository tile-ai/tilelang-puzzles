"""
Puzzle 03: Outer Vector Add
==============
In this puzzle we will enter the 2D world!

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from utils import test_puzzle, bench_puzzle

"""
Consider an outer vector addition operation. The result is a matrix where
each element (i, j) is the sum of A[i] and B[j].

The main difference from the previous puzzle is that C is now a 2D tensor and
we have two different iterators in buffers A and B. So the dataflow is also
a little different.

But remeber that any N dimensional tensor can be viewed as a 1D tensor in memory.
So we just need to handle the indexing properly.

03-1: Outer vector addition.

Inputs:
    A: [N,]  # input tensor
    B: [M,]  # input tensor
    N: int   # size of the tensor. 1 <= N <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    C: [N, M]  # output tensor

Definition:
    for i in range(N):
        for j in range(M):
            C[i, j] = A[i] + B[j]
"""

def ref_outer_add(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int, M: int,dtype: torch.dtype):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.shape[0] == N
    assert B.shape[0] == M
    assert len(C.shape) == 2
    assert C.shape[0] == N
    assert C.shape[1] == M
    assert dtype == A.dtype == B.dtype == C.dtype == torch.float16
    torch.add(input=A[:, None], other=B[None, :], out=C)


@tilelang.jit
def tl_outer_add(N: int, M: int, dtype: torch.dtype, BLOCK_N: int, BLOCK_M: int):
    @T.prim_func
    def kernel(
        A: T.Buffer((N,), dtype),
        B: T.Buffer((M,), dtype),
        C: T.Buffer((N, M), dtype)
    ):
        # TODO: Implement this function
        with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=256) as (bx, by):
            n_idx = bx * BLOCK_N
            m_idx = by * BLOCK_M
            A_local = T.alloc_fragment((BLOCK_N,), dtype)
            B_local = T.alloc_fragment((BLOCK_M,), dtype)
            C_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)

            T.copy(A[n_idx], A_local)
            T.copy(B[m_idx], B_local)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                C_local[i, j] = A_local[i] + B_local[j]
            T.copy(C_local, C[n_idx, m_idx])

    return kernel


def run_outer_add():
    print("\n=== Outer Vector Add ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 1024
    BLOCK_M = 1024
    dtype = torch.float16
    test_puzzle(tl_outer_add, ref_outer_add, {"N": N, "M": M, "dtype": dtype}, {"BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M})


if __name__ == "__main__":
    run_outer_add()
