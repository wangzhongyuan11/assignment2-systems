import torch
import triton
import triton.language as tl
from einops import rearrange


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        k_pos = tl.arange(0, K_TILE_SIZE) + i * K_TILE_SIZE
        mask = (q_pos[:, None] < N_QUERIES) & (k_pos[None, :] < N_KEYS)
        if is_causal:
            mask = (q_pos[:, None] >= k_pos[None, :]) & mask
        S_ij = tl.where(mask, S_ij, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(S_ij, axis=-1))
        P_ij = tl.exp(S_ij - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        m_i = m_ij
        l_i = alpha * l_i + tl.sum(P_ij, axis=-1)
        P_ij = P_ij.to(V_j.dtype)
        O_i *= alpha[:, None]
        O_i = tl.dot(P_ij, V_j, acc=O_i)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    O_i = (O_i / (l_i[:, None] + 1e-6)).to(O_block_ptr.type.element_ty)
    l_i = (m_i + tl.log(l_i)).to(L_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_i, boundary_check=(0,))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))

@triton.jit
def flash_bwd_preprocess(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERIES, D,
    Q_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_TILE = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        O = tl.load(O_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        D_TILE += tl.sum(O * dO, axis=-1)
        O_block_ptr = O_block_ptr.advance((0, D_TILE_SIZE))
        dO_block_ptr = dO_block_ptr.advance((0, D_TILE_SIZE))
    tl.store(D_block_ptr, D_TILE, boundary_check=(0,))

@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, L_ptr, dO_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    dO_i = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")
    L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    q_start = query_tile_index * Q_TILE_SIZE
    q_pos = q_start + tl.arange(0, Q_TILE_SIZE)
    if is_causal:
        j_stop = tl.cdiv(q_start + Q_TILE_SIZE, K_TILE_SIZE)
    else:
        j_stop = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(j_stop):
        K_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            k_pos = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :]
            S_ij = tl.where(mask, S_ij, -1.0e6)
        P_ij = tl.exp(S_ij - L_i[:, None])
        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale
        dS_ij = dS_ij.to(K_j.dtype)
        dQ_i = tl.dot(dS_ij, K_j, acc=dQ_i)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, dQ_i.to(dQ_block_ptr.type.element_ty), boundary_check=(0,))

@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, L_ptr, dO_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    k_start = key_tile_index * K_TILE_SIZE
    k_pos = k_start + tl.arange(0, K_TILE_SIZE)
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        should_compute = True
        if is_causal:
            if (i + 1) * Q_TILE_SIZE <= k_start:
                should_compute = False
        if should_compute:
            Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
            dO_i = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")
            L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
            S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
            if is_causal:
                q_pos = tl.arange(0, Q_TILE_SIZE) + i * Q_TILE_SIZE
                mask = q_pos[:, None] >= k_pos[None, :]
                S_ij = tl.where(mask, S_ij, -1.0e6)
            P_ij = tl.exp(S_ij - L_i[:, None])
            P_ij = P_ij.to(dO_i.dtype)
            dV_j = tl.dot(tl.trans(P_ij), dO_i, acc=dV_j)
            dP_ij = tl.dot(dO_i, tl.trans(V_j))
            dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale
            dS_ij = dS_ij.to(Q_i.dtype)
            dK_j = tl.dot(tl.trans(dS_ij), Q_i, acc=dK_j)
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty), boundary_check=(0,))

class FlashAttnWithTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 分块大小
        Bq = 32
        Bk = 32
        input_shape = Q.shape
        Q = rearrange(Q, "... q d -> (...) q d")
        K = rearrange(K, "... k d -> (...) k d")
        V = rearrange(V, "... k d -> (...) k d")
        batch_size, N_QUERIES, d = Q.shape
        N_KEYS = K.shape[-2]
        scale = 1.0 / (d ** 0.5)
        Tq = (N_QUERIES + Bq - 1) // Bq
        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device)
        flash_fwd_kernel[(Tq, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale, D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
            is_causal = is_causal
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.Bq = Bq
        ctx.Bk = Bk
        ctx.is_causal = is_causal
        return O.view(input_shape)

    @staticmethod
    def backward(ctx, dO):
        # 分块大小
        Bq = 32
        Bk = 32
        Bd = 8
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch_size, N_QUERIES, d = Q.shape
        N_KEYS = K.shape[-2]
        scale = 1.0 / (d ** 0.5)
        Tq = (N_QUERIES + Bq - 1) // Bq
        Tk = (N_KEYS + Bk - 1) // Bk
        D = torch.zeros_like(L)
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        flash_bwd_preprocess[(Tq, batch_size)](
            O, dO, D,
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            N_QUERIES=N_QUERIES, D=d,
            Q_TILE_SIZE=Bq,
            D_TILE_SIZE=Bd
        )
        flash_bwd_dq_kernel[(Tq, batch_size)](
            Q, K, V,
            D, L, dO, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal
        )
        flash_bwd_dkdv_kernel[(Tk, batch_size)](
            Q, K, V,
            D, L, dO, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=scale,
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal
        )
        return dQ, dK, dV, None