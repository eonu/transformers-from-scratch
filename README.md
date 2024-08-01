# Transformer From Scratch

The repository contains a modular Python implementation of the transformer architecture for natural language understanding tasks, according to:

- The seminal paper _Attention Is All You Need_ by Vaswani et al.<sup><a href="#references">[1]</a></sup> that details the novel attention-based transformer architecture and its application to sequence-to-sequence tasks, demonstrating its effectiveness by achieving state-of-the-art performance in machine translation, surpassing previous LSTM and CNN based neural machine translation architectures.
- The chapter on _Transformers and Large Language Models_ from _Speech and Language Processing_ by Jurafsky & Martin<sup><a href="#references">[2]</a></sup> which provides a more comprehensive and graphical look into some of the high-level details discussed in _Attention Is All You Need_.

## Restrictions

This project is implemented using [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

### PyTorch restrictions

As PyTorch provides a number of transformer and attention related layers in its [`torch.nn`](https://pytorch.org/docs/stable/nn.html) submodule, this project explicitly avoids the use of:

- [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention)
- [`torch.nn.Transformer`](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer)
- [`torch.nn.TransformerEncoder`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder)/[`torch.nn.TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer)
- [`torch.nn.TransformerDecoder`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder)/[`torch.nn.TransformerDecoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer)
- [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)

All other layers provided by `torch.nn` are allowed, including:

- [`nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding): For token embedding look-up by vocabulary ID.
- [`nn.LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm): For layer normalization as implemented in _Attention Is All You Need_.

### Other restrictions

- Transformer models implemented and made available in other libraries such as HuggingFace's [`transformers`](https://huggingface.co/docs/transformers/en/index) are not used in this project.
- However, the tokenizers provided by `transformers` were used, as developing tokenization algorithms was not the primary objective of this project.
- Finally no existing _"x from scratch"_ resources were used, such as the famous _Let's build GPT: from scratch, in code, spelled out._ by Andrej Karpathy<sup><a href="#references">[3]</a></sup>.

## Details

While the original architecture implemented in _Attention Is All You Need_ is an encoder-decoder based architecture using transformers for neural machine translation which is a sequence-to-sequence learning task, this projected was designed to be more general, allowing for a variety of natural language tasks by implementing encoder-only, decoder-only and encoder-decoder architectures.

<table>
    <tbody>
        <tr>
            <td></td>
            <td><b>Encoder-only</b></td>
            <td><b>Decoder-only</b></td>
            <td><b>Encoder-decoder</b></td>
        </tr>
        <tr>
            <td><b>Diagram</b></td>
            <td><img src="assets/encoder-only.svg"/></td>
            <td><img src="assets/decoder-only.svg"/></td>
            <td><img src="assets/encoder-decoder.svg"/></td>
        </tr>
        <tr>
            <td><b>Tasks</b></td>
            <td>Inference</td>
            <td>Autoregressive generation</td>
            <td>Sequence-to-sequence generation</td>
        </tr>
        <tr>
            <td><b>Example use-cases</b></td>
            <td>
                <ul>
                    <li>Sentiment classification/regression</li>
                    <li>Intent classification</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>Text generation</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>Neural machine translation</li>
                    <li>Text summarization</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

## Models and datasets

TODO

## Repository structure

TODO

## Installation

TODO

## Running

TODO

## References

<table>
    <tbody>
        <tr>
            <td>[1]</td>
            <td>
            <a href="https://dl.acm.org/doi/10.5555/3295222.3295349">Vaswani et al., <b>"Attention Is All You Need"</b>, <em>Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017)</em>, 6000-6010.</a>
            </td>
        </tr>
        <tr>
            <td>[2]</td>
            <td>
            <a href="https://web.stanford.edu/~jurafsky/slp3/10.pdf">Dan Jurafsky & James H. Martin, <b>"Transformers and Large Language Models"</b>, <em>Speech and Language Processing, 3rd ed. draft (2024)</em>, ch. 10.</a>
            </td>
        </tr>
        <tr>
            <td>[3]</td>
            <td>
            <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Andrej Karpathy <b>"Let's build GPT: from scratch, in code, spelled out."</b>, <em>YouTube (2023)</em></a>
            </td>
        </tr>
    </tbody>
</table>
