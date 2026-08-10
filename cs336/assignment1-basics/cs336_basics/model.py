import torch.nn as nn
import torch
from math import sqrt


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
