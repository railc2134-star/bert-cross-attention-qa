BERT Cross-Attention Question Answering Model

This project implements a sequence-to-sequence model using a BERT encoder and a Transformer decoder with cross-attention.

The model is trained on the SQuAD dataset to generate answers given a question.

Overview

The system uses:
- A pretrained BERT model as the encoder
- A Transformer decoder with self-attention and cross-attention
- Cross-attention layers that attend to encoded question representations

The goal is to generate answer tokens conditioned on a question input.

Dataset

The model is trained on the SQuAD dataset which contains:
- Question text
- Corresponding answer text spans

Each answer is tokenized and shifted for next-token prediction.

Model Architecture

Encoder:
- Pretrained bert-base-multilingual-cased
- Outputs contextual embeddings of the question

Decoder:
- Token embedding layer
- Positional embeddings
- Multi-layer Transformer decoder blocks
- Each block contains:
  - Masked self-attention
  - Cross-attention over encoder outputs
  - Feedforward network
  - Layer normalization

Training

The model is trained using:
- Cross entropy loss
- Adam optimizer

Teacher forcing is used for decoder inputs.

The decoder learns to predict the next token in the answer sequence.

Cross-Attention

Cross-attention allows the decoder to focus on relevant parts of the encoded question while generating each token of the answer.

This enables context-aware generation conditioned on input questions.

Limitations

This implementation is experimental and has limitations:

- No answer span extraction supervision
- Heavy memory usage due to full BERT encoder
- Precomputed encoder outputs reduce adaptability
- Limited training epochs due to computational cost
- No beam search decoding during inference
