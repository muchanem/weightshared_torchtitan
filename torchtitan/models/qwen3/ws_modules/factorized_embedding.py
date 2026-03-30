# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Factorized embeddings for parameter-efficient language models.

This module implements factorized embeddings where the embedding matrix is decomposed
as: vocab_size x dim -> vocab_size x d_emb @ d_emb x dim

This reduces embedding parameters from vocab_size * dim to vocab_size * d_emb + d_emb * dim,
which can be significant for large vocabularies.

The FactorizedOutput class uses F.linear with transposed weights to enable weight tying
without the meta device issues that can occur with nn.Linear weight assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common import Embedding, Linear
from torchtitan.protocols.module import Module


class FactorizedEmbedding(Module):
    """Factorized token embedding layer.

    Decomposes the embedding matrix into two smaller matrices:
    - tok_embeddings: vocab_size x d_emb (lookup table)
    - tok_embeddings_up: d_emb x dim (projection layer)

    Forward pass: tokens -> embedding lookup -> linear projection -> hidden states

    Args:
        vocab_size: Size of the vocabulary.
        d_emb: Intermediate embedding dimension.
        dim: Final model dimension.

    Example:
        >>> emb = FactorizedEmbedding(vocab_size=32000, d_emb=256, dim=2048)
        >>> tokens = torch.randint(0, 32000, (2, 128))
        >>> hidden = emb(tokens)  # shape: (2, 128, 2048)
    """

    def __init__(self, vocab_size: int, d_emb: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.dim = dim

        self.tok_embeddings = Embedding.Config().build(
            num_embeddings=vocab_size, embedding_dim=d_emb
        )
        self.tok_embeddings_up = Linear.Config().build(
            in_features=d_emb, out_features=dim
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert token indices to hidden states.

        Args:
            tokens: Token indices of shape (batch_size, seq_len).

        Returns:
            Hidden states of shape (batch_size, seq_len, dim).
        """
        emb = self.tok_embeddings(tokens)
        return self.tok_embeddings_up(emb)

    def init_weights(self, **kwargs) -> None:
        """Initialize embedding weights.

        Scales the embedding matrix by 1/sqrt(d_emb) so that the tied output
        path (h @ P.weight @ E.weight.T) produces logits with the correct
        variance. Without this scaling, the sum over d_emb elements in the
        final matmul amplifies logit variance by a factor of d_emb.
        """
        self.tok_embeddings.init_weights()
        self.tok_embeddings_up.init_weights()
        with torch.no_grad():
            self.tok_embeddings.weight.div_(self.d_emb**0.5)


class FactorizedOutput(Module):
    """Factorized output projection with weight tying to embeddings.

    Uses F.linear with transposed embedding weights to project hidden states
    back to vocabulary logits. This enables weight tying without the meta device
    issues that can occur with direct nn.Linear weight assignment.

    The computation is:
        h -> F.linear(h, tok_embeddings_up.weight) -> intermediate
        intermediate -> F.linear(intermediate, tok_embeddings.weight) -> logits

    This is equivalent to:
        h @ tok_embeddings_up.weight.T @ tok_embeddings.weight.T

    Args:
        embedding: The FactorizedEmbedding to tie weights with.

    Example:
        >>> emb = FactorizedEmbedding(vocab_size=32000, d_emb=256, dim=2048)
        >>> output = FactorizedOutput(emb)
        >>> hidden = torch.randn(2, 128, 2048)
        >>> logits = output(hidden)  # shape: (2, 128, 32000)
    """

    def __init__(self, embedding: FactorizedEmbedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            h: Hidden states of shape (batch_size, seq_len, dim).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        # Project down: tok_embeddings_up.weight has shape (dim, d_emb)
        # We need (batch, seq, dim) @ (dim, d_emb) -> (batch, seq, d_emb)
        # Using matmul directly since F.linear would transpose the wrong way
        h_down = torch.matmul(h, self.embedding.tok_embeddings_up.weight)
        # Project to vocab: tok_embeddings.weight has shape (vocab, d_emb)
        # We need (batch, seq, d_emb) @ (d_emb, vocab) -> (batch, seq, vocab)
        return torch.matmul(h_down, self.embedding.tok_embeddings.weight.t())


