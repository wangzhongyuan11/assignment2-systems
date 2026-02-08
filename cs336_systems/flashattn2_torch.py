import torch
import math
from einops import einsum

def flash_backward(Q, K, V, O, L, dO, is_causal = False):
    D = (O * dO).sum(dim=-1)
    Bq = 32
    Bk = 32
    Nq, d = Q.shape[-2], Q.shape[-1]
    Nk = K.shape[-2]
    Tq = (Nq + Bq - 1) // Bq
    Tk = (Nk + Bk - 1) // Bk
    scale = 1.0 / math.sqrt(d)
    dQ = torch.zeros_like(Q, dtype=torch.float32)
    dK = torch.zeros_like(K, dtype=torch.float32)
    dV = torch.zeros_like(V, dtype=torch.float32)
    all_q_pos = torch.arange(Nq, device=Q.device)
    all_k_pos = torch.arange(Nk, device=Q.device)
    for j in range(Tk):
        start_k = j * Bk
        end_k = min((j + 1) * Bk, Nk)
        Kj = K[..., start_k:end_k, :]
        Vj = V[..., start_k:end_k, :]
        dKj = torch.zeros_like(Kj, dtype=torch.float32)
        dVj = torch.zeros_like(Vj, dtype=torch.float32)
        k_pos = all_k_pos[start_k:end_k]
        for i in range(Tq):
            start_q = i * Bq
            end_q = min((i + 1) * Bq, Nq)
            Li = L[..., start_q:end_q]
            Di = D[..., start_q:end_q]
            Qi = Q[..., start_q:end_q, :]
            dQi = dQ[..., start_q:end_q, :]
            dOi = dO[..., start_q:end_q, :]
            Sij = einsum(Qi, Kj, "... q d, ... k d -> ... q k") * scale
            q_pos = all_q_pos[start_q:end_q]
            if is_causal:
                mask = q_pos[:, None] >= k_pos[None, :]
                Sij = Sij.where(mask, torch.tensor(-1.0e6, dtype=Sij.dtype, device=Sij.device))
            Pij = torch.exp(Sij - Li[..., None])
            Pij = Pij.to(dOi.dtype)
            dVj += einsum(Pij, dOi, "... q k, ... q d -> ... k d")
            dPij = einsum(dOi, Vj, "... q d, ... k d -> ... q k")
            dSij = Pij * (dPij - Di[..., None]) * scale
            dQi += einsum(dSij, Kj, "... q k, ... k d -> ... q d")
            dQ[..., start_q:end_q, :] = dQi
            dKj += einsum(dSij, Qi, "... q k, ... q d -> ... k d")
        dK[..., start_k:end_k, :] = dKj
        dV[..., start_k:end_k, :] = dVj
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None

compiled_backward = torch.compile(flash_backward)

class FlashAttnWithTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 分块大小
        Bq = 32
        Bk = 32
        Nq, d = Q.shape[-2],Q.shape[-1]
        Nk = K.shape[-2]
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk
        scale = 1.0 / math.sqrt(d)
        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device)
        all_q_pos = torch.arange(Nq, device=Q.device)
        all_k_pos = torch.arange(Nk, device=Q.device)
        for i in range(Tq):
            start_q = i * Bq
            end_q = min((i + 1) * Bq, Nq)
            Qi= Q[..., start_q:end_q, :]
            Oi = torch.zeros_like(Qi)
            mi = torch.full(Qi.shape[:-1], float('-inf'), device=Q.device)
            Li = torch.zeros_like(mi)
            q_pos = all_q_pos[start_q:end_q]
            for j in range(Tk):
                start_k = j * Bk
                end_k = min((j + 1) * Bk, Nk)
                Kj = K[..., start_k:end_k, :]
                Vj = V[..., start_k:end_k, :]
                Sij = einsum(Qi, Kj, "... q d, ... k d -> ... q k") * scale
                k_pos = all_k_pos[start_k:end_k]
                if is_causal:
                    mask = q_pos[:, None] >= k_pos[None, :]
                    Sij = Sij.where(mask, torch.tensor(-1.0e6, dtype=Sij.dtype, device=Sij.device))
                mij = torch.maximum(mi, torch.max(Sij, dim=-1)[0])
                Pij = torch.exp(Sij - mij.unsqueeze(-1))
                s = torch.exp(mi - mij)
                mi = mij
                Li = s * Li + Pij.sum(dim=-1)
                Pij = Pij.to(Vj.dtype)
                Oi = Oi * s.unsqueeze(-1) + einsum(Pij, Vj, "... q k, ... k d -> ... q d")
            O[..., start_q:end_q, :] = (Oi / (Li.unsqueeze(-1) + 1e-6)).to(Q.dtype)
            L[..., start_q:end_q] = (mi + torch.log(Li)).to(L.dtype)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        return compiled_backward(Q, K, V, O, L, dO, is_causal)


