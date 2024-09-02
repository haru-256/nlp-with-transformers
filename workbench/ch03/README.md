# Full-Scratch Transfomer(IMDB Classification)

## Description

## Dataset

Transfomer is trained by [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb).

> Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.

This dataset label is binary 0/1,  0: negative and 1: positive.
Because this dataset is not imbalance, random model accuracy has 50%.

## Model

We use Transfomer Encoder. Model Summary is the following.

```txt
   | Name                                   | Type                    | Params | Mode
--------------------------------------------------------------------------------------------

0  | encoder                                | TransformerEncoder      | 1.5 M  | train
1  | encoder.embeddings                     | Embeddings              | 1.3 M  | train
2  | encoder.embeddings.token_embeddings    | Embedding               | 1.3 M  | train
3  | encoder.embeddings.position_embeddings | Embedding               | 65.7 K | train
4  | encoder.embeddings.layer_norm          | LayerNorm               | 256    | train
5  | encoder.embeddings.dropout             | Dropout                 | 0      | train
6  | encoder.blocks                         | ModuleList              | 198 K  | train
7  | encoder.blocks.0                       | TransformerEncoderBlock | 198 K  | train
8  | encoder.blocks.0.layer_norm_1          | LayerNorm               | 256    | train
9  | encoder.blocks.0.layer_norm_2          | LayerNorm               | 256    | train
10 | encoder.blocks.0.self_attention        | MultiHeadSelfAttention  | 66.0 K | train
11 | encoder.blocks.0.feed_forward          | PointwiseFeedForward    | 131 K  | train
12 | classifier                             | Classifier              | 129    | train
13 | classifier.classifier                  | Sequential              | 129    | train
14 | classifier.classifier.0                | Dropout                 | 0      | train
15 | classifier.classifier.1                | Linear                  | 129    | train
16 | loss_fn                                | BCEWithLogitsLoss       | 0      | train
17 | accuracy                               | BinaryAccuracy          | 0      | train
--------------------------------------------------------------------------------------------
```
