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
