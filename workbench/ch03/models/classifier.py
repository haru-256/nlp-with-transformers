import torch
import torch.nn as nn

from utils import create_self_attention_mask

from .module import TransformerEncoder


class TransformerForSequenceClassification(nn.Module):
    def __init__(
        self,
        pad_idx: int,
        vocab_size: int,
        hidden_size: int,
        num_encoder_blocks: int,
        num_attention_heads: int,
        hidden_dropout_prob: float,
        output_dropout_prob: float,
        max_position_embeddings: int,
        num_labels: int,
    ):
        """Transformer model for sequence classification

        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_encoder_blocks=num_encoder_blocks,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
        )
        self.dropout = nn.Dropout(output_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor):
        """Forward pass for transformer model

        Args:
            x: id tensor, shape (batch_size, seq_len)

        Returns:
            output tensor, shape (batch_size, num_labels)
        """
        # attention mask for PAD token
        attn_mask = create_self_attention_mask(x, self.pad_idx, is_causal=False)
        # encoder output shape is (batch_size, seq_len, hidden_size)
        # extract the first element of the sequence (the [CLS] token), shape (batch_size, hidden_size)
        enc = self.encoder(x, attn_mask)[:, 0, :]
        h = self.dropout(enc)
        # shape (batch_size, num_labels)
        out = self.classifier(h)
        return out
