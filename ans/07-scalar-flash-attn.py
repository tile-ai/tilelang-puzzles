"""
Puzzle 07: Scalar FlashAttention
==============
From softmax to FlashAttention, we just need some computation.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from utils import test_puzzle, bench_puzzle

"""
Now we have conquered softmax / online softmax, we can now implement one of the most important operator in LLMs: FlashAttention.

To ensure a progressive learning experience, we will implement a scalar version of FlashAttention. And we also remove the multi-head attention part. So in total we only have two dimensions: batch size B and sequence length S, which are aligned with N, M in the previous puzzle. After such simplification, you will find we are not so far from the FlashAttention algorithm. And with TileLang, we can easily extend it to the full FlashAttention.

06-1: Simplified Scalar Flash Attention.

Inputs:
    Q: [B, S]  # input tensor
    K: [B, S]  # input tensor
    V: [B, S]  # input tensor
    B: int   # batch size dimension. 1 <= B <= 256
    S: int   # sequence length dimension. 1 <= S <= 16384
    dtype: torch.dtype  # data type of the tensor. e.g., torch.float32, torch.int32, etc.

Output:
    O: [B, S]  # output tensor

Intermediates:
    MAX: dtype  # max value of each row
    SUM: dtype  # summation of each row
    QK: [B, S]  # results of q*k
    P:  [B, S]  # results of softmax(q*k) (not divided by summation).

Definition:
    for i in range(B):
        SUM = 0
        MAX = -inf
        for j in range(S):
            QK[i, j] = Q[i, j] * K[i, j]
            MAX = max(QK[i, j], MAX)
        for j in range(S):
            P[i, j] = exp(QK[i, j] - MAX)
            SUM += P[i, j]
        for j in range(M):
            O[i, j] = P[i, j] / SUM * V[i, j]
"""

def ref_scalar_flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor, B: int, S: int, dtype: torch.dtype):
    assert len(Q.shape) == 2
    assert len(K.shape) == 2
    assert len(V.shape) == 2
    assert len(O.shape) == 2
    assert Q.shape[0] == K.shape[0] == V.shape[0] == O.shape[0] == B
    assert Q.shape[1] == K.shape[1] == V.shape[1] == O.shape[1] == S
    assert dtype == Q.dtype == K.dtype == V.dtype == O.dtype == torch.float32

    P = torch.softmax(Q * K, dim=1)
    torch.mul(P, V, out=O)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_scalar_flash_attn(B: int, S: int, dtype: torch.dtype, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504

    @T.prim_func
    def kernel(
        Q: T.Buffer((B, S), dtype),
        K: T.Buffer((B, S), dtype),
        V: T.Buffer((B, S), dtype),
        O: T.Buffer((B, S), dtype),
    ):
        # TODO: Implement this function
        with T.Kernel(B // BLOCK_B, threads=256) as pid_b:
            Q_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
            K_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
            V_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
            O_local = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)

            cur_QK = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
            cur_exp_QK = T.alloc_fragment([BLOCK_B, BLOCK_S], dtype)
            cur_max_QK = T.alloc_fragment([BLOCK_B], dtype)
            cur_sum_exp_QK = T.alloc_fragment([BLOCK_B], dtype)

            lse = T.alloc_fragment([BLOCK_B], dtype)

            T.fill(lse, -T.infinity(dtype))

            # The first loop use an online algorithm to compute LSE.
            for s_blk_id in T.Serial(S // BLOCK_S):
                T.copy(Q[pid_b * BLOCK_B, s_blk_id * BLOCK_S], Q_local)
                T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)

                for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                    cur_QK[i, j] = Q_local[i, j] * K_local[i, j]

                T.reduce_max(cur_QK, cur_max_QK, dim=1, clear=True)

                for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                    cur_exp_QK[i, j] = T.exp2(cur_QK[i, j] * log2_e - cur_max_QK[i] * log2_e)

                T.reduce_sum(cur_exp_QK, cur_sum_exp_QK, dim=1, clear=True)

                for i in T.Parallel(BLOCK_B):
                    lse[i] = cur_max_QK[i] * log2_e + T.log2(T.exp2(lse[i] - cur_max_QK[i] * log2_e) + cur_sum_exp_QK[i])

            # The second loop use LSE to get the final output.
            # TODO(chaofan): Now this implementation is not very efficient.
            for s_blk_id in T.Serial(S // BLOCK_S):
                T.copy(Q[pid_b * BLOCK_B, s_blk_id * BLOCK_S], Q_local)
                T.copy(K[pid_b * BLOCK_B, s_blk_id * BLOCK_S], K_local)
                T.copy(V[pid_b * BLOCK_B, s_blk_id * BLOCK_S], V_local)

                for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                    O_local[i, j] = T.exp2(Q_local[i, j] * K_local[i, j] * log2_e - lse[i]) * V_local[i, j]

                T.copy(O_local, O[pid_b * BLOCK_B, s_blk_id * BLOCK_S])

    return kernel


def run_scalar_flash_attn():
    print("\n=== Scalar Flash Attention ===\n")
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    dtype = torch.float32
    test_puzzle(tl_scalar_flash_attn, ref_scalar_flash_attn, {"B": B, "S": S, "dtype": dtype}, {"BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S})
    bench_puzzle(tl_scalar_flash_attn, ref_scalar_flash_attn, {"B": B, "S": S, "dtype": dtype}, {"BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S}, bench_torch=True)

if __name__ == "__main__":
    run_scalar_flash_attn()
