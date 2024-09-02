import logging
import pathlib

import lightning as L

from data.dataset import IMDbDataModule, SpecialTokens
from models.classifier import TransformerForSequenceClassification
from utils.logging import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def main():
    save_dir = pathlib.Path("data/data")

    datamodule = IMDbDataModule(save_dir=save_dir, batch_size=128)
    datamodule.prepare_data()
    reversed_vocab = {v: k for k, v in datamodule.vocab.items()}
    vocab_size = len(datamodule.vocab)

    model = TransformerForSequenceClassification(
        pad_idx=reversed_vocab[SpecialTokens.PAD],
        vocab_size=vocab_size,
        hidden_size=128,
        num_encoder_blocks=1,
        num_attention_heads=1,
        hidden_dropout_prob=0.5,
        output_dropout_prob=0.5,
        max_position_embeddings=datamodule.max_seq_len + 1,  # +1 for the <cls> token
        num_labels=2,
        cls_token_pos=reversed_vocab[SpecialTokens.CLS],
        learning_rate=1e-3,
    )

    trainer = L.Trainer(fast_dev_run=10, max_epochs=10, accelerator="cpu")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
