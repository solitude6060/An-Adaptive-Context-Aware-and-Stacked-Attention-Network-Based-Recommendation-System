# Adaptive Context-Aware Recommendation with Stacked Attention

Official implementation of **An Adaptive, Context-Aware, and Stacked Attention Network-Based Recommendation System to Capture Users’ Temporal Preference**.

**Jung-Hsien Chiang, Chung-Yao Ma, Chi-Shiang Wang, and Pei-Yi Hao**  
*IEEE Transactions on Knowledge and Data Engineering*, 35(4), 3404–3418, 2023.

[Paper](https://doi.org/10.1109/TKDE.2022.3140387) · [IEEE Xplore](https://ieeexplore.ieee.org/document/9670723) · [PDF](Adaptive_Context-Aware_Recommendation_System_via_Stacked_Attention_Network_IEEE_TKDE.pdf) · [University publication record](https://researchoutput.ncku.edu.tw/en/publications/an-adaptive-context-aware-and-stacked-attention-network-based-rec/) · [Citation](#citation)

The implementation originated with Chung-Yao Ma’s 2020 master’s thesis at National Cheng Kung University, *Adaptive Context-Aware Recommendation System via Stacked Attention Network*. [Thesis record](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22108NCKU5392046%22.&searchmode=basic).

## Method

The model builds a context representation from recent user interactions:

- **Contextual item attention** weights items within each user’s history.
- **Stacked multi-head user attention** adapts the user representation using that context.
- Optional positional encoding and convolutional modules incorporate temporal information.

Training compares the scores of an observed item and a sampled unobserved item. The corrected training loss is `-logsigmoid(positive_score - negative_score).mean()`.

## Implementation update

The September 2026 correction normalizes contextual item attention over the history dimension in all three attention paths and uses numerically stable negative log-sigmoid pairwise loss. It also replaces the unsupported `Tensor.copy()` call with `Tensor.clone()` in the stacked residual path.

Historical code used batch-axis contextual normalization and `1 - sigmoid(score_difference)`. The correction changes training and scoring behavior; old checkpoints and stored results must be identified by their original revision. This update does not certify reproduction of every published result. The original dataset files, pretrained weights, and historical result files are unchanged.

## Installation

The original environment used **Python 3.6**, **PyTorch 1.4.0**, and an NVIDIA Tesla V100. Exact historical dependencies are in [`requirements.txt`](requirements.txt).

```bash
git clone https://github.com/solitude6060/An-Adaptive-Context-Aware-and-Stacked-Attention-Network-Based-Recommendation-System.git
cd An-Adaptive-Context-Aware-and-Stacked-Attention-Network-Based-Recommendation-System
python -m pip install -r requirements.txt
```

Use an environment compatible with those legacy package versions. The original training entry point requires CUDA. The correction's focused CPU tests were also checked with Python 3.11 and PyTorch 2.5.1; this is not a claim that the full legacy training pipeline has been migrated to those versions.

## Data

Processed files are provided for MovieLens-100K (`ml-100k`), MovieLens-1M (`ml-1m`), Pinterest (`pinterest`), and Amazon Beauty (`beauty`).

```text
dataset/<dataset>/processed_data/
├── user_session.pkl
└── testing_data.pkl
model/bpr/<dataset>/       # Existing BPR pretrained weights
model/models/<dataset>/    # Saved recommendation models
result/<dataset>/          # Historical training and evaluation logs
```

- `user_session.pkl`: `{user_id: [item_id, ...]}` ordered interaction sequences.
- `testing_data.pkl`: `{user_id: [negative_1, ..., negative_99, test_item]}` fixed evaluation candidates.

Evaluation ranks one held-out item against 99 sampled unobserved items per user. These are sampled-candidate results; they do not represent full-catalogue ranking. Load pickle and model files only from trusted sources and follow the original datasets' usage terms.

## Training

Example for the thesis's five-head, three-stack configuration without extra temporal modules, using the existing item-pretraining option:

```bash
python main.py -D ml-100k -d 32 -l 0.001 -e 50 -b 256 -w 5 -n 10 -i 1 -s 3 -h 5 -t 0
```

The CLI default is five stacks; `-s 3` selects three explicitly. This command specifies a configuration, not a guarantee of matching the historical reported metrics.

| Flag | Meaning | Default |
| --- | --- | --- |
| `-D` | Dataset identifier | `ml-100k` |
| `-d` | Embedding dimension | 32 |
| `-l` | Learning rate | 0.001 |
| `-e` | Training epochs | 50 |
| `-b` | Batch size in positive/negative pairs | 256 |
| `-w` | Context window length | 5 |
| `-n` | Training negatives per context | 10 |
| `-i` | Load pretrained item embeddings | 1 |
| `-s` | Stacked user-attention modules | 5 |
| `-h` | Heads per module | 5 |
| `-t` | Temporal mode | 0 |

Temporal modes: `0` none; `1` positional encoding; `2` convolution; `3` convolution then positional encoding; `4` positional encoding then convolution; `5` separate positional/convolutional representations followed by concatenation and a fully connected layer.

### Legacy execution notes

- The CLI's `-i 0` path currently references an uninitialized `bpr_item_weight`; the command above uses `-i 1`. The model constructor itself supports random item initialization.
- The active constructor loads pretrained **item** embeddings; user embeddings are randomly initialized. Report this behavior explicitly when comparing with descriptions that pretrain both.
- The original script evaluates during training and saves according to test metrics. For new research, use an independent validation split for model selection and reserve test data for final evaluation.
- Full training, other temporal variants, and all datasets have not been rerun as part of this source correction.

## Regression tests

In an environment with PyTorch and pytest installed:

```bash
python -m pytest -q -p no:cacheprovider tests/test_thesis_math.py
```

The tests cover history-axis normalization, batch/chunk consistency with user BatchNorm disabled, the positional/convolutional path, and stable loss values and gradients. They run on CPU without launching training.

## Citation

If you use this implementation in your research, please cite the journal article:

```bibtex
@article{chiang2023adaptive,
  author  = {Chiang, Jung-Hsien and Ma, Chung-Yao and Wang, Chi-Shiang and Hao, Pei-Yi},
  title   = {An Adaptive, Context-Aware, and Stacked Attention Network-Based Recommendation System to Capture Users' Temporal Preference},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2023},
  volume  = {35},
  number  = {4},
  pages   = {3404--3418},
  doi     = {10.1109/TKDE.2022.3140387}
}
```

For work specifically referencing the master's thesis:

```bibtex
@mastersthesis{ma2020adaptive,
  author = {Ma, Chung-Yao},
  title  = {Adaptive Context-Aware Recommendation System via Stacked Attention Network},
  school = {National Cheng Kung University},
  year   = {2020},
  url    = {https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22108NCKU5392046%22.&searchmode=basic}
}
```

GitHub's **Cite this repository** action uses [`CITATION.cff`](CITATION.cff). Please include the code revision and any implementation changes when reporting experiments.

## Contact

Implementation: **Chung-Yao Ma** · [chungyao.ma@gmail.com](mailto:chungyao.ma@gmail.com). Please use GitHub Issues for reproducible code questions.
