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

from common.utils import test_puzzle, bench_puzzle

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

def ref_dequant_matmul(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 2
    assert len(B.shape) == 2
    assert A.shape[1] == B.shape[0] # K
    assert A.dtype == torch.float16
    assert B.dtype == torch.uint8

    K = A.shape[1]
    N = B.shape[1] * 2 # B.shape[1] == N // 2 because of packing

    B_dequantized = torch.zeros((K, N), dtype=torch.float16, device=B.device)
    B_dequantized[:, ::2] = B[:, :] & 0x0F
    B_dequantized[:, 1::2] = (B[:, :] >> 4) & 0x0F
    B_dequantized = B_dequantized.to(torch.float16) - 8.0 # dequantize

    return torch.matmul(input=A, other=B_dequantized)


@tilelang.jit
def tl_dequant_matmul(A, B, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
    M, N, K = T.const("M, N, K")
    A_dtype = T.float16
    B_storage_dtype = T.uint8
    A: T.Tensor((M, K), A_dtype)
    B: T.Tensor((K, N//2), B_storage_dtype)
    C = T.empty((M, N), A_dtype)
    accum_dtype = T.float32

    # TODO: Implement this function

    return C


def run_dequant_matmul():
    print("\n=== Dequantized Matrix Multiplication ===\n")

    M = 4096
    N = 4096
    K = 4096
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    # A_dtype = torch.float16
    # B_storage_dtype = torch.uint8
    # accum_dtype = torch.float32
    test_puzzle(tl_dequant_matmul, ref_dequant_matmul, {"M": M, "N": N, "K": K, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K})


if __name__ == "__main__":
    run_dequant_matmul()
