"""
Puzzle 10: Dequantized Matrix Multiplication
==============
In the final puzzle in our journey, let's build a very useful variant of matmul kernel which can be used in real research work.

Category: ["official"]
Difficulty: ["hard"]
"""

import tilelang
import tilelang.language as T
import torch

from utils import test_puzzle, bench_puzzle

"""
Dequantized Matrix Multiplication is to multiply two matrices in different precisons, which is widely
used in the depolyment of quantized LLMs. We consider a common setting here: FP16A * INT4B. Because
INT4 is less than a byte, we usually packed two INT4 in a storage type, like UINT8.

10-1: Dequantized Matrix Multiplication.

Inputs:
    A: [M, K]  # input tensor
    B: [K, N]  # input tensor
    N: int   # size of the tensor. 1 <= N <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192
    A_dtype: torch.dtype  # data type of the A/C tensor, high precison.
    B_storage_dtype: torch.dtype  # storage type of the B tensor, low precison.

Output:
    C: [M, N]  # output tensor

Intermediates:
    B_high: [1, ] # high bits of B
    B_low: [1,]   # low bits of B

Definition:
    for i in range(M):
        for j in range(N // 2):
            C[i, j * 2] = 0
            C[i, j * 2 + 1] = 0
            for k in range(K):
                B_low = A_dtype(B[k, j] & 0x0F) - 8.0  # signed int4
                B_high = A_dtype((B[k, j] >> 4) & 0x0F) - 8.0  # signed int4
                C[i, j * 2] += A[i, k] * B_low
                C[i, j * 2 + 1] += A[i, k] * B_high
"""

def ref_dequant_matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, A_dtype: torch.dtype, B_storage_dtype: torch.dtype, accum_dtype: torch.dtype):
    assert len(A.shape) == 2
    assert len(B.shape) == 2
    assert len(C.shape) == 2
    assert A.shape[0] == C.shape[0] == M
    assert A.shape[1] == B.shape[0] == K
    assert B.shape[1] == N // 2 # packed
    assert C.shape[1] == N
    assert A_dtype == A.dtype == C.dtype == torch.float16
    assert B_storage_dtype == torch.uint8
    assert accum_dtype == torch.float32

    B_dequantized = torch.zeros((K, N), dtype=torch.float16, device=B.device)
    B_dequantized[:, ::2] = B[:, :] & 0x0F
    B_dequantized[:, 1::2] = (B[:, :] >> 4) & 0x0F
    B_dequantized = B_dequantized.to(torch.float16) - 8.0 # dequantize

    torch.matmul(input=A, other=B_dequantized, out=C)


@tilelang.jit
def tl_dequant_matmul(M: int, N: int, K: int, A_dtype, B_storage_dtype, accum_dtype, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), A_dtype),
        B: T.Buffer((K, N//2), B_storage_dtype),
        C: T.Buffer((M, N), A_dtype),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (pid_m, pid_n):
            # TODO: Implement this function
            pass

    return kernel


def run_dequant_matmul():
    print("\n=== Dequantized Matrix Multiplication ===\n")

    M = 4096
    N = 4096
    K = 4096

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    A_dtype = torch.float16
    B_storage_dtype = torch.uint8
    accum_dtype = torch.float32
    test_puzzle(tl_dequant_matmul, ref_dequant_matmul, {"M": M, "N": N, "K": K, "A_dtype": A_dtype, "B_storage_dtype": B_storage_dtype, "accum_dtype": accum_dtype}, {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K})


if __name__ == "__main__":
    run_dequant_matmul()
