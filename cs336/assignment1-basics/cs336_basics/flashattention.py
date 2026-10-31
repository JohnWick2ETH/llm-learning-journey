import torch
import math
import triton
import triton.language as tl


class FlashAttentionTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        # implement the forward pass of vanilla attention
        # using the flash attention algorithm, but using only
        # standard PyTorch operations (no Triton!)
        assert len(q.shape) == 3
        assert len(k.shape) == 3
        assert len(v.shape) == 3

        tile_size = 32

        num_q_tiles = (q.shape[-2] + tile_size - 1) // tile_size
        num_k_tiles = (k.shape[-2] + tile_size - 1) // tile_size

        Out = torch.empty((q.shape[0], q.shape[1], v.shape[-1]), device=q.device, dtype=v.dtype)
        Logsumexp = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=v.dtype)

        for i in range(num_q_tiles):
            q_start = i * tile_size
            q_end = min((i + 1) * tile_size, q.shape[-2])

            q_tile = q[..., q_start:q_end, :]

            # init Mi = -inf, Li = 0, and Oi = 0 for each tile
            m = torch.full(
                (q.shape[0], q_end - q_start), float("-inf"), device=q.device, dtype=q.dtype
            )
            l = torch.zeros((q.shape[0], q_end - q_start), device=q.device, dtype=q.dtype)
            o = torch.zeros(
                (q.shape[0], q_end - q_start, v.shape[-1]), device=q.device, dtype=v.dtype
            )

            for j in range(num_k_tiles):
                start = j * tile_size
                end = min((j + 1) * tile_size, k.shape[-2])
                k_tile = k[..., start:end, :]
                v_tile = v[..., start:end, :]

                # a. compute local tile's scaled attention score matrix Sij = (qi * kj^T) / sqrt(d)
                Sij = torch.einsum("... q d, ... k d -> ... q k", q_tile, k_tile) / (
                    math.sqrt(q.shape[-1])
                )

                # b. compute local tile's row-wise max of Sij up to j: Mi_old = Mi, Mi = max(Mi, row_max(Sij))
                Mi_old = m
                m = torch.max(m, torch.max(Sij, dim=-1).values)

                # m is 2d tensor
                assert len(m.shape) == 2

                assert (
                    Mi_old.shape == m.shape
                ), f"Mi_old shape = {Mi_old.shape}, m shape = {m.shape}, q_start = {q_start}, q_end = {q_end}"
                alpha = torch.exp(Mi_old - m)
                adjusted_exp_sij = torch.exp(Sij - m.unsqueeze(-1))

                # c. local tile's denominator: Lij = exp(Mi_old - Mi) * Li^{j-1} + row_sum(exp(Sij - Mi^{j}))
                Lij = alpha * l + torch.sum(adjusted_exp_sij, dim=-1).squeeze(-1)
                l = Lij

                # d. local tile's output: Oij = exp(Mi_old - Mi) * Oi^{j-1} + exp(Sij - Mi^{j}) * vj
                Oij = alpha.unsqueeze(-1) * o + torch.einsum(
                    "... q k, ... k d -> ... q d", adjusted_exp_sij, v_tile
                )
                o = Oij

            # 2. Oi = diag(Li)^{-1} * Oi and write it to output tensor's memory
            Out[..., q_start:q_end, :] = o / l.unsqueeze(-1)
            Logsumexp[..., q_start:q_end] = m + torch.log(l)

        ctx.save_for_backward(q, k, v, Out, Logsumexp)

        # 3. return tensor O
        return Out

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()


# TODO: handle causal attention
@triton.jit
def flash_attention_fwd_kernel(
    # fmt: off
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    # fmt: on
):
    #########################################################################
    ####  implement the forward pass of flash attention using Triton kernels
    #########################################################################

    # Program indices
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

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_start = i * K_TILE_SIZE
        # load k, v
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # compute local attention score matrix: Sij = q * ki^T / sqrt(d) of size O(TILE_SIZE^2)
        attn_score = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        # for causal masking, valid k indices are those that are less than N_KEYS
        # and less than or equal to the current query index (q_offsets)
        k_offsets = k_start + tl.arange(0, K_TILE_SIZE)
        if is_causal:
            valid_k = (k_offsets[None, :] < N_KEYS) & (k_offsets[None, :] <= q_offsets[:, None])
        else:
            valid_k = k_offsets[None, :] < N_KEYS
        attn_score = tl.where(valid_k, attn_score, float("-inf"))

        # compute Mi' = max(Mi, row_max(Sij))
        m_new = tl.maximum(m, tl.max(attn_score, axis=1))

        # adjusted_exp_sij = exp(Sij - Mi')
        adjusted_exp_s = tl.exp(attn_score - m_new[:, None])

        # alpha = exp(Mi - Mi')
        alpha = tl.exp(m - m_new)

        # L' = alpha * L + row_sum(adjusted_exp_sij)
        l = alpha * l + tl.sum(adjusted_exp_s, axis=1)

        # O' = alpha * O + adjusted_exp_sij * vj
        p = adjusted_exp_s.to(v.dtype)
        o = alpha[:, None] * o + tl.dot(p, v, out_dtype=tl.float32)

        # advance k_block
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        m = m_new

    # 2. Oi = diag(Li)^{-1} * Oi and write it to output tensor's memory
    o = o / l[:, None]
    final_l = tl.log(l) + m

    tl.store(L_block_ptr, final_l, boundary_check=(0,))
    tl.store(O_block_ptr, o, boundary_check=(0, 1))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        #######################################################################
        #### implement the forward pass of flash attention using Triton kernels
        #######################################################################

        # assumes q, k, v of shape (b, n, d)
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        assert k.shape[-2] == v.shape[-2]
        assert q.device == k.device == v.device
        assert q.is_cuda, "Expected CUDA Tensors"
        assert q.dtype == k.dtype == v.dtype
        assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)

        D, N_QUERIES, N_KEYS = q.shape[-1], q.shape[-2], k.shape[-2]
        assert N_QUERIES > 0 and N_KEYS > 0 and D > 0

        Out = torch.empty_like(q)
        Logsumexp = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        # TODO: make this configurable
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        flash_attention_fwd_kernel[(triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), q.shape[0])](
            # fmt: off
            q, k, v, Out, Logsumexp,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            Logsumexp.stride(0), Logsumexp.stride(1),
            N_QUERIES, N_KEYS, 1 / math.sqrt(D), D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal,
            # fmt: on
        )

        ctx.save_for_backward(q, k, v, Out, Logsumexp)

        return Out

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()
