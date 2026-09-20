import torch.nn as nn
import torch
from math import sqrt, cos, sin, pow
from .utils import softmax
from einops import einsum, rearrange


class Linear(nn.Module):
    """
    returns W*x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        sigma = sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            tensor=weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma
        )

        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns W*x
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """
    convert token id to a vector in the vector space of dimension d_model
    that is, token id is mapped to a vector in R^d_model.

    the embedding matrix/tensor has shape (vocab_size, d_model)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(tensor=weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns embedding vectors for token id tensor.
        We assume that the token id tensor has shape (batch_size, sequence_length)
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # starting at 1 makes per-feature scale initially neutral
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        computes RMS norm for each vector in the input tensor of shape (batch_size, sequence_length, d_model)
        returns a new tensor with same shape as input
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # rms is a tensor with shape (batch_size, sequence_length, 1)
        rms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

        w_aligned = self.weight.reshape(1, 1, x.shape[-1])

        f = lambda a, s, g: a * g / torch.sqrt(s * s / self.d_model + self.eps)

        y = f(x, rms, w_aligned)

        return y.to(in_dtype)


class SwiGLUFeedForwardNetwork(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        computes W2 * (SiLU(W1*x) ⊙ W3*x)
        """
        silu = lambda x: torch.sigmoid(x) * x

        y0 = self.w1(x)
        y1 = silu(y0)
        y2 = self.w3(x)

        return self.w2(y1 * y2)


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        softmax(masked(Q * K^T / sqrt(d_k))) * V
        """

        # q is tensor with shape (... queries d_k)
        # k is tensor with shape (... keys d_k)
        # v is tensor with shape (... keys d_v)
        # mask is tensor with shape (... queries keys)

        qk = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")

        dk = q.size(-1)
        qk = qk / sqrt(dk)

        # tensor with shape (... queries keys)
        masked_qk = qk.masked_fill(~mask, float("-inf"))
        alpha = softmax(masked_qk, -1)

        # returned tensor is of shape (... queries d_v)
        return einsum(alpha, v, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        hi = SingleHeadSelfAttention(qi, ki, vi) where [qi] = W_Q*x, [ki] = W_K*x, [vi] = W_V*x
             and x is a tensor with shape (... seq_len d_model)
        h = h0 || h1 || ...
        returns o*h
        """
        seq_len = x.size(-2)

        # x has shape (... seq_len d_model)
        # each qi has shape (... seq_len d_k) where d_k = d_model / num_heads
        # TODO: reduce three matrix multiplies to one.
        qi = rearrange(
            self.q_proj(x),
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        ki = rearrange(
            self.k_proj(x),
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        vi = rearrange(
            self.v_proj(x),
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
        )

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=bool)
        )
        att = Attention()

        # the tensor has shape (... num_heads seq_len d_v)
        heads = att(qi, ki, vi, causal_mask)
        heads = rearrange(
            heads, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
        )

        return self.output_proj(heads)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # precompute 2x2 diagonal matrix M_{i,k} from 𝜃_{i,k}
        # 𝜃_{i,k} = i / 𝜃^((2k-2)/d) for k ∈ {1, ..., d/2}

        running_thetas = [1.0]
        for k in range(1, d_k // 2):
            running_thetas.append(running_thetas[k - 1] * pow(theta, 2.0 / d_k))

        m = []
        m.append([[[1.0, 0.0], [0.0, 1.0]] for _ in range(d_k // 2)])
        for i in range(1, max_seq_len):
            mi = []
            for theta_k in running_thetas:
                c, s = cos(i / theta_k), sin(i / theta_k)
                mi.append([[c, -s], [s, c]])
            m.append(mi)

        tensor_m = torch.Tensor(m).to(device, dtype)
        assert tensor_m.shape[0] == max_seq_len
        assert tensor_m.shape[1] == d_k // 2
        assert tensor_m.shape[2] == 2
        assert tensor_m.shape[3] == 2

        self.register_buffer("rotation_matrices", tensor_m, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x is a tensor of shape (... seq_len d_k)
        token positions are a tensor of shape (... seq_len)
        """
        # note that self.matrix is a tensor of shape (seq_len, d_k/2, 2, 2)
        # for each pos, apply rotation transformation to the input embedding vector
        rotations = self.rotation_matrices[token_positions]
        x_pairs = rearrange(
            x, "... seq_len (half pair) -> ... seq_len half pair", pair=2
        )
        rotated = einsum(rotations, x_pairs, "... s p o i, ... s p i -> ... s p o")

        return rearrange(
            rotated, "... seq_len half pair -> ... seq_len (half pair)", pair=2
        )


class MultiHeadSelfAttentionWithROPE(MultiHeadSelfAttention):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(d_model, num_heads, device, dtype)
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):

        seq_len = x.size(-2)

        # x has shape (... seq_len d_model)
        # each qi has shape (... seq_len d_k) where d_k = d_model / num_heads
        # TODO: reduce three matrix multiplies to one.
        qi = rearrange(
            self.q_proj(x),
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        token_positions.unsqueeze_(1)
        qi = self.rope(qi, token_positions)
        ki = rearrange(
            self.k_proj(x),
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        ki = self.rope(ki, token_positions)
        vi = rearrange(
            self.v_proj(x),
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
        )

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=bool)
        )
        att = Attention()

        # the tensor has shape (... num_heads seq_len d_v)
        heads = att(qi, ki, vi, causal_mask)
        heads = rearrange(
            heads, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
        )

        return self.output_proj(heads)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFeedForwardNetwork(d_model, d_ff, device, dtype)
        self.attn = MultiHeadSelfAttentionWithROPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a tensor of shape (... seq_len d_model)
        seq_len = x.size(-2)

        token_positions = torch.arange(
            seq_len,
            device=x.device,
            dtype=torch.long,
        ).expand(*x.shape[:-1])

        t1 = self.ln1(x)
        t2 = self.attn(t1, token_positions)
        y = x + t2

        t3 = self.ln2(y)
        t4 = self.ffn(t3)
        y = y + t4

        return y


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lm_head = Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype
        )
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, in_tokens: torch.Tensor) -> torch.Tensor:
        """
        run transformer forward pass on input tokens which are int tensor of shape
           (batch_size, seq_len) to get unnormalized next-word distribution for each token.

        returns a float tensor of shape (batch_size, sequence_length, vocab_size)
        """
        # input embedding
        x = self.token_embeddings(in_tokens)

        # the result of input embedding is a tensor of shape (batch_size, seq_len, d_model)
        # run `num_layers` transformer blocks iteratively on x
        for transformer_block in self.layers:
            x = transformer_block(x)

        # final norm
        y = self.ln_final(x)

        # output unembedding to get tensor of shape (batch_size, seq_len, vocab_size)
        return self.lm_head(y)
