<p align="center">
  <h1 align="center">Transformers From Scratch</h1>
</p>

<p align="center">
  <sup>
    <b>Contents</b>:&nbsp;
    <a href="#restrictions">Restrictions</a> ·
    <a href="#details">Details</a> ·
    <a href="#datasets">Datasets</a> ·
    <a href="#models-and-examples">Models and examples</a> ·
    <a href="#repository-structure">Repository structure</a> ·
    <a href="#installation">Installation</a> ·
    <a href="#running">Running</a> ·
    <a href="#references">References</a>
  </sup>
</p>

The repository contains a modular Python implementation of transformer architectures for natural language understanding tasks, according to:

- The seminal paper _Attention Is All You Need_ by Vaswani et al.<sup><a href="#references">[1]</a></sup> that details the novel attention-based transformer architecture and its application to sequence-to-sequence tasks, demonstrating its effectiveness by achieving state-of-the-art performance in machine translation, surpassing previous LSTM and CNN based neural machine translation architectures.
- The chapter on _Transformers and Large Language Models_ from _Speech and Language Processing_ by Jurafsky & Martin<sup><a href="#references">[2]</a></sup> which provides a more comprehensive and illustrative look into some of the high-level details discussed in _Attention Is All You Need_.

## Restrictions

This project is implemented using [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

### PyTorch restrictions

As PyTorch provides a number of transformer and attention related layers in its [`torch.nn`](https://pytorch.org/docs/stable/nn.html) submodule, this project explicitly avoids the use of:

- [`torch.nn.Transformer`](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer)
- [`torch.nn.TransformerEncoder`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder)/[`torch.nn.TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer)
- [`torch.nn.TransformerDecoder`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html#torch.nn.TransformerDecoder)/[`torch.nn.TransformerDecoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html#torch.nn.TransformerDecoderLayer)
- [`torch.nn.MultiHeadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention)
- [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)

All other layers provided by `torch.nn` are allowed, including:

- [`nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding): For token embedding look-up by vocabulary ID.
- [`nn.LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm): For layer normalization as implemented in _Attention Is All You Need_.

### Other restrictions

- Transformer models implemented and made available in other libraries such as HuggingFace's [`transformers`](https://huggingface.co/docs/transformers/en/index) are not used in this project.
- However, the tokenizers provided by `transformers` were used, as developing tokenization algorithms was not the primary objective of this project.
- No existing _"x from scratch"_ resources were used, such as the famous _Let's build GPT: from scratch, in code, spelled out._ by Andrej Karpathy<sup><a href="#references">[3]</a></sup>.
- No other online resources were used, apart from official documentation for packages such as [PyTorch](https://pytorch.org/docs/stable/index.html), [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Huggingface Tokenizers](https://huggingface.co/docs/transformers/en/main_classes/tokenizer).

## Details

While the original architecture described in _Attention Is All You Need_ is an encoder-decoder based architecture using transformers for neural machine translation which is a sequence-to-sequence learning task, this project was designed to be more general, allowing for a variety of natural language tasks by implementing encoder-only, decoder-only and encoder-decoder architectures.

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
            <td>Contextual embedding and supervised inference</td>
            <td>Autoregressive generation</td>
            <td>Sequence-to-sequence generation</td>
        </tr>
        <tr>
            <td><b>Example use-cases</b></td>
            <td>
                <ul>
                    <li>Producing contextual token embeddings</li>
                    <li>Sentiment classification</li>
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
                    <li>Machine translation</li>
                    <li>Text summarization</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

## Datasets

The following datasets were used to test the above transformer implementations on various tasks.

- [arXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts): arXiv manuscripts and their metadata including titles, abstracts and categories.
- [CommonLit Readability Prize](https://www.kaggle.com/competitions/commonlitreadabilityprize): Literary passages and their associated "readability" score for use in grade 3-12 classrooms.
- [Reddit r/FloridaMan](https://www.kaggle.com/datasets/bcruise/reddit-rfloridaman): News headlines about various (often funny and irrational) actions performed by Florida men and women.
- [Europarl](https://www.kaggle.com/datasets/nltkdata/europarl): Transcriptions of European Parliament proceedings between 1996-2006, collected in 11 languages.

## Models and examples

### Encoder-only models

- [`ClassifierLM`](transformer/models/classifier.py): A generic transformer-based language model for assigning classes to text.
  - [`notebooks/arxiv_categorization.ipynb`](notebooks/arxiv_categorization.ipynb) applies this model to the _arXiv Paper Abstracts_ dataset to categorize arXiv manuscripts based on their titles.
- [`RegressorLM`](transformer/models/regressor.py): A generic transformer-based language model for assigning scores to text.
  - [`notebooks/commonlit_readability.ipynb`](notebooks/commonlit_readability.ipynb) applies this model to the _CommonLit Readability Prize_ dataset to rate the complexity of literary passages for grade 3-12 students.

### Decoder-only models

- [`CausalLM`](transformer/models/causal.py): A generic transformer-based language model for generating text in an autoregressive manner.
  - [`notebooks/florida_man_generation.ipynb`](notebooks/florida_man.ipynb) applies this model to the _Reddit r/FloridaMan_ dataset to generate humorous news headlines involving the (mis)adventures of Florida men and women.

### Encoder-decoder models

- [`Seq2SeqLM`](transformer/models/seq2seq.py): A generic transformer-based language model for generating output text given an input text.
  - [`notebooks/arxiv_summarization.ipynb`](notebooks/arxiv_summarization.ipynb) applies this model to the _arxiv Paper Abstracts_ dataset to generate arXiv paper titles by summarizing their corresponding abstracts.
  - [`notebooks/europarl_translation.ipynb`](notebooks/europarl_translation.ipynb) applies this model to the _Europarl_ dataset to translate transcribed parliamentiary proceedings from French to English.

## Repository structure

- [**`notebooks/`**](notebooks/): Notebooks applying the models in [`transformer.models`](transformer/models/) to various datasets.
- [**`transformer/`**](transformer/): Core package containing the transformer implementations.
  - [**`dataloaders/`**](transformer/dataloaders/): [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)s for each model in [`transformer.models`](transformer/models/).
  - [**`models/`**](transformer/models/): Task-specific transformers implemented using [`transformer.modules.transformers`](transformer/modules/transformers/).
  - [**`modules/`**](transformer/modules/): [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)s used within the transformers in [`transformer.models`](transformer/models/).
    - [**`transformers/`**](transformer/modules/transformers/): Encoder-only, decoder-only and encoder-decoder transformer definitions.
    - [`attention.py`](transformer/modules/attention.py): Masked/unmasked multi-head self attention definition.
    - [`block.py`](transformer/modules/block.py): Transformer block definition.
    - [`embedding.py`](transformer/modules/embedding.py): Positional encoding and input embedding definition.
  - [**`utils/`**](transformer/utils/): Supporting custom layers, functions and constants.
  - [`params.py`](transformer/params.py): Pydantic hyper-parameter classes for modules in [`transformer.modules`](transformer/modules/).

## Installation

The transformer implementation is installable as a local Python package, named `transformer`.

```console
pip install -e .
```

To run the notebooks, you will need additional dependencies which can be installed with the `notebooks` extra.

```console
pip install -e ".[notebooks]"
```

**This package was developed on Python 3.11.8, so it is recommended to use a virtual environment with the same version.**

## Running

You should be able to simply run the Jupyter notebooks in the [`notebooks/`](notebooks/) folder.

_Beware, they take time – even with a good GPU (especially the sequence-to-sequence ones)!_

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

---

<p align="center">
  &copy; 2024-2025, Edwin Onuonga - Published under the terms of the <a href="https://opensource.org/licenses/MIT">MIT</a> license.<br/>
  <em>Authored and maintained by Edwin Onuonga.</em>
</p>
