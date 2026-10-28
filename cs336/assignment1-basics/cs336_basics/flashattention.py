import torch
import einops
import math


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

        Out = torch.empty(
            (q.shape[0], q.shape[1], v.shape[-1]), device=q.device, dtype=v.dtype
        )
        Logsumexp = torch.empty(
            (q.shape[0], q.shape[1]), device=q.device, dtype=v.dtype
        )

        for i in range(num_q_tiles):
            q_start = i * tile_size
            q_end = min((i + 1) * tile_size, q.shape[-2])

            q_tile = q[..., q_start:q_end, :]

            # init Mi = -inf, Li = 0, and Oi = 0 for each tile
            m = torch.full(
                (q.shape[0], q_end - q_start),
                float("-inf"),
                device=q.device,
                dtype=q.dtype,
            )
            l = torch.zeros(
                (q.shape[0], q_end - q_start), device=q.device, dtype=q.dtype
            )
            o = torch.zeros(
                (q.shape[0], q_end - q_start, v.shape[-1]),
                device=q.device,
                dtype=v.dtype,
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


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        # implement the forward pass of flash attention using Triton kernels
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass
