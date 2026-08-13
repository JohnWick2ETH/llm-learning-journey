import torch.nn as nn
import torch
from math import sqrt, cos, sin, pow
from .utils import softmax
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        weight = torch.empty(out_features, in_features)
        sigma = sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            tensor=weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma
        )

        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns W*x
        """
        return x @ self.weight.T


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        matrix = torch.empty(num_embeddings, embedding_dim)
        nn.init.trunc_normal_(tensor=matrix, mean=0.0, std=1.0, a=-3.0, b=3.0)

        self.e_matrix = nn.Parameter(matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        returns embedding vectors for token ids with shape (batch_size, sequence_length)
        """
        # returned tensor has shape (batch_size, sequence_length, d_model)

        return self.e_matrix[token_ids]


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
        self.weights = nn.Parameter(torch.empty(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        computes RMS norm for each vector in the input tensor of shape (batch_size, sequence_length, d_model)
        returns a new tensor with same shape as input
        """
        assert self.weights != None

        in_dtype = x.dtype
        x = x.to(torch.float32)

        # rms is a tensor with shape (batch_size, sequence_length, 1)
        rms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

        w_aligned = self.weights.reshape(1, 1, x.shape[2])

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

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        computes W2 * (SiLU(W1*x) ⊙ W3*x)
        """
        silu = lambda x: torch.sigmoid(x) * x

        y1 = silu(x @ self.w1.T)
        y2 = x @ self.w3.T

        return (y1 * y2) @ self.w2.T


class SingleHeadSelfAttension(nn.Module):

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

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.k_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.v_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.o_weight = nn.Parameter(torch.empty(d_model, d_model))

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
        qi = rearrange(
            x @ self.q_weight.T,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        ki = rearrange(
            x @ self.k_weight.T,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        vi = rearrange(
            x @ self.v_weight.T,
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
        )

        casual_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
        att = SingleHeadSelfAttension()

        # the tensor has shape (... num_heads seq_len d_v)
        heads = att(qi, ki, vi, casual_mask)
        heads = rearrange(
            heads, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
        )

        return heads @ self.o_weight.T


class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.seq_len = max_seq_len

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

        tensor_m = torch.Tensor(m)
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
