"""
Puzzle 05: Reduce Sum
==============
In this puzzle, you will learn how to do reduce in TileLang.

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from utils import test_puzzle, bench_puzzle

"""
We alreadly do broadcasting in previous example. Now let's see how to do reduction. Luckily,
we don't need to implement detailed reduction logics since TileLang provides built-in
TileOps. Before this, T.copy is the only TileOp we have seen. But we have experienced that
with T.copy and T.Parallel we can already do many things!

HINT:
1. For reduction, we have `T.reduce` and `T.reduce_xxx`, where xxx represents the reduction
operation, e.g., `T.reduce_sum`. Note that for efficiency, we need to perform these TileOps in the fragment buffers instead of global memory.
2. You may need a serial loop to do this puzzle. Use `T.Serial` to create a serial loop.
3. For numerical stability, we shift the data type to float32 for now.

05-1: Reduce sum.

Inputs:
    A: [N, M]  # input tensor
    B: [M,]  # input tensor
    N: int   # size of the tensor. 1 <= N <= 4096
    M: int   # size of the tensor. 1 <= M <= 16384
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    B: [N,]  # output tensor

Definition:
    for i in range(N):
        B[i] = 0
        for j in range(M):
            B[i] += A[i, j]
"""

def ref_reduce_sum(A: torch.Tensor, B: torch.Tensor, N: int, M: int, dtype: torch.dtype):
    assert len(A.shape) == 2
    assert len(B.shape) == 1
    assert A.shape[0] == B.shape[0] == N
    assert A.shape[1] == M
    assert dtype == A.dtype == B.dtype == torch.float32

    B.copy_(torch.sum(A, dim=1))


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_reduce_sum(N: int, M: int, dtype: torch.dtype, BLOCK_N: int, BLOCK_M: int):
    @T.prim_func
    def kernel(
        A: T.Buffer((N, M), dtype),
        B: T.Buffer((N,), dtype),
    ):
        # TODO: Implement this function
        pass

    return kernel


def run_reduce_sum():
    print("\n=== Reduce Sum ===\n")
    N = 4096
    M = 16384
    BLOCK_N = 16
    BLOCK_M = 128
    dtype = torch.float32
    test_puzzle(tl_reduce_sum, ref_reduce_sum, {"N": N, "M": M, "dtype": dtype}, {"BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M})
    bench_puzzle(tl_reduce_sum, ref_reduce_sum, {"N": N, "M": M, "dtype": dtype}, {"BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M}, bench_torch=True)

if __name__ == "__main__":
    run_reduce_sum()
