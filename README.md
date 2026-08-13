<div align="center">

# 🎸 MyGO!!!!!

### Modality-incomplete Fake News Video Detection via Prompt-assisted Modality Disentangling Model

<p><i>「迷子でもいい、前へ進め。」</i><br/>
<sub>It's okay to be lost — keep moving forward.</sub></p>

[![Paper](https://img.shields.io/badge/Paper-ACM%20TOMM%202025-b31b1b?style=flat-square&logo=acm)](https://doi.org/10.1145/3785481)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3785481-blue?style=flat-square)](https://doi.org/10.1145/3785481)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#-license)
[![Members](https://img.shields.io/badge/!!!!!-5%20members%20%7C%205%20missing%20rates-ff69b4?style=flat-square)](#-why-mygo)

**Mingjie Qiu** · **Zhiyi Tan** · **Bing-Kun Bao**<sup>†</sup>
<br/>
*Nanjing University of Posts and Telecommunications*

<sub><sup>†</sup> Corresponding author</sub>

<p>
  <a href="#-overview">Overview</a> ·
  <a href="#-why-mygo">Why "MyGO"?</a> ·
  <a href="#-highlights">Highlights</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-dataset-fakesv">Dataset</a> ·
  <a href="#-usage">Usage</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 📖 Overview

Short-video platforms have become a major channel of misinformation. Yet almost every existing
fake news video detector silently assumes that **video, audio and text are all available** — an
assumption that breaks down in the wild, where modalities go missing because of privacy settings,
technical limitations and editing preferences.

**MyGO** (Modality Disentan**G**lement with Missing Pr**O**mpt) is the first study that explores
*incomplete multimodal learning* for fake news video detection. It attacks the problem at two
different levels:

| Level | Problem in prior work | MyGO's answer |
|:--|:--|:--|
| **Representation** | Highly coupled cross-modal fusion buries intra- and inter-modality dependencies | **CKA** refines each modality with caption-guided keyframe attention; **MDN** disentangles it into shared + specific parts |
| **Training** | The optimiser over-fits *strong* modality combinations, under-fits *weak* ones | **PMA** mines weak combinations with a prompt global memory and rebalances the loss |

> On the newly built **FakeSV+** benchmark, MyGO improves accuracy by **3.79 % – 4.85 %** on average
> over state-of-the-art baselines, and the margin *widens* as the missing rate grows.

---

## 🎵 Why "MyGO"?

<div align="center">
<i>「一人だけじゃ、バンドはできないんだよ。」</i><br/>
<sub>You can't make a band all by yourself. — and you can't detect fake news with a single modality either.</sub>
</div>

<br/>

The acronym is a genuine one — **M**odality Disentan**G**lement with Missing Pr**O**mpt — but the name
was chosen because it is also the band from *BanG Dream! It's MyGO!!!!!*, and the coincidence runs
deeper than it looks:

- **「MyGO」 is homophonous with 迷子 (*maigo*, "a lost child")**. A modality-incomplete video is
  exactly that: a sample that has *lost* part of itself and no longer knows where it belongs in the
  feature space. Our whole paper is about making such lost samples find their way.
- **The five exclamation marks stand for the five band members.** They also happen to match the
  five missing-rate settings we evaluate: **η ∈ {0, 0.1, 0.3, 0.5, 0.7}**.
- **MyGO!!!!! is the first band of the series built around twin guitars.** CKA is likewise built
  around a **two-stream** co-attention rather than a single fused stream.
- **The band re-forms out of a broken past instead of pretending nothing is missing.** That is the
  opposite of imputation-based methods, which try to hallucinate the missing modality back; MyGO
  instead learns to *play with whoever is on stage tonight*.

<details>
<summary><b>🎤 Band ↔ module mapping (purely for fun — click to expand)</b></summary>

<br/>

| Member | Role on stage | Counterpart in the model |
|:--|:--|:--|
| **Tomori Takamatsu** 高松灯 | Vocalist who writes the lyrics in her notebook first | The **embedded captions** of CKA — the words come first and tell everyone else which moments matter |
| **Anon Chihaya** 千早愛音 | Guitar; the one who reaches out and connects people | Stream ➊ of the **prompted co-attention**, `CKA(V, T)` |
| **Raana Kaname** 要楽奈 | Guitar; plays whatever she feels, gloriously off the leash | Stream ➋, `CKA(A, T)` — the **modality-specific** part that refuses to be averaged away |
| **Soyo Nagasaki** 長崎そよ | Bass; smiles on the surface, holds a very different tune underneath | The **shared / specific decomposition** with its orthogonal constraint `L_o` |
| **Taki Shiina** 椎名立希 | Drums; keeps the tempo and calls out whoever is slacking | The **PMA loss regularizer** — it finds the under-fitted combinations and makes them practise harder |
| **CRYCHIC** (the band before) | The past that fell apart | The **event-level contrastive loss** `L_ctrs`, which pulls videos of the same event back together |

</details>

---

## ✨ Highlights

<table>
<tr>
<td width="33%" valign="top">

### 🎯 CKA
**Caption-guided Keyframe Attention**

<sub>♪ *the vocalist's notebook*</sub>

Embedded captions tell us *which frames matter*. CKA turns them into a frame-level weight
distribution and prompts a two-stream co-attention with the missing prompt, so noisy
transitions/outtakes are suppressed and the available modalities get the spotlight.

*Robust by design*: when text is missing the weight degenerates into a uniform distribution
determined by a learnable bias.

</td>
<td width="33%" valign="top">

### 🧩 MDN
**Modality Disentangling Network**

<sub>♪ *everyone plays their own line*</sub>

Each refined feature is split into a **shared** part (consistency across modalities, further
shaped by an *event-level* supervised contrastive loss) and a **specific** part (complementary
cues, kept orthogonal to the shared subspace).

Pairwise transformer encoders then measure cross-modality *inconsistency* — a strong signal for
fabricated content.

</td>
<td width="33%" valign="top">

### 🧠 PMA
**Prompt-assisted Modality Aligning**

<sub>♪ *the drummer counts everyone in*</sub>

A 4-digit **missing prompt** explicitly marks the modality combination of every video. PMA keeps a
**zoned, dynamically weighted global memory** of the combinations that repeatedly incur high loss,
then re-weights their loss so they stop being under-fitted.

*Plug-and-play*: +0.04 % → +3.33 % accuracy when attached to TKCM / TwtrD / SVFEND.

</td>
</tr>
</table>

---

## 🏗 Architecture

```
                       ┌──────────────── Data Preprocessing Stage ────────────────┐
  raw short video ───► │ frame-level features (VGG19 / VGGish / BERT)             │
                       │ zero padding for missing modalities            (Eq. 1)   │
                       │ missing prompt encoding  P = [any, V, T, A]    (Eq. 2)   │
                       └──────────────────────────┬──────────────────────────────┘
                                                  ▼
   ┌──────────────────────────────── MyGO Stage ─────────────────────────────────┐
   │                                                                             │
   │  ┌── CKA (Sec. 3.2) ──────────┐   ┌── MDN (Sec. 3.3) ──────────────────┐    │
   │  │ caption weight  U   (Eq. 3)│   │ context-aware shared encoder H^s   │    │
   │  │ prompted co-attn    (Eq. 4)│──►│                        (Eq. 5, 6)  │    │
   │  │      ↓                     │   │ specific decoupling W^u_*  (Eq. 7) │    │
   │  │  F_v   F_t   F_a           │   │ orthogonal constraint L_o  (Eq. 8) │    │
   │  └────────────────────────────┘   │ single gate + pairwise  (Eq. 9,10) │    │
   │                                   │ inter-modality dependency Z (Eq.11)│    │
   │                                   └───────────────┬────────────────────┘    │
   │                                                   ▼                         │
   │                                        MLP classifier  (Eq. 12)             │
   │                                                   │                         │
   │  ┌── PMA (Sec. 3.4) ─────────────────────────────┐ │                         │
   │  │ prompt global memory: warm-up / inactive /    │ │                         │
   │  │ active zones with dynamic weights             │◄┘                         │
   │  │ loss regularizer  L'_cls = u·[C]/M ∘ L_cls    │                           │
   │  └───────────────────────┬───────────────────────┘                           │
   └──────────────────────────┼───────────────────────────────────────────────────┘
                              ▼
             L = L'_cls  +  α · L_ctrs  +  β · L_o          (Eq. 14)
```

**Missing prompt template** (Sec. 3.1) — four binary digits, `1` marks *missing*:

| digit | 1st | 2nd | 3rd | 4th |
|:--|:--|:--|:--|:--|
| meaning | any modality missing | video missing | text missing | audio missing |

> Example: a video whose **audio** is missing → `1001`; a modality-complete video → `0000`.
>
> <sub>♪ Think of it as the setlist note before a live: *who is on stage tonight?* `0000` is the full
> band; `1001` means the drums are out and the rest have to carry the song anyway.</sub>

---

## 📂 Repository Structure

```
MyGO/
├── requirements.txt
└── src/
    ├── main.py                     # CLI entry, exposes every paper hyper-parameter
    ├── dataset_mask.py             # build FakeSV+ from FakeSV (Eq. 15, Table 2)
    ├── common/
    │   ├── abstract.py             # AbstractModel + epoch hooks used by PMA
    │   ├── function.py             # co-attention, padding and other primitives
    │   └── trainer.py              # training loop, early stop, modality-wise eval
    ├── configs/
    │   ├── overall.yaml            # global training / evaluation settings
    │   ├── dataset/FakeSV*.yaml    # one file per missing rate η
    │   └── model/MyGO.yaml         # MyGO hyper-parameters + ablation switches
    ├── models/
    │   ├── mygo.py                 # ★ the MyGO model
    │   ├── svfend.py               # SVFEND baseline (+ optional PMA)
    │   ├── simplefusion.py         # concatenation-fusion baseline
    │   └── modules/
    │       ├── cka.py              # ★ Caption-guided Keyframe Attention
    │       ├── mdn.py              # ★ Modality Disentangling Network
    │       └── pma.py              # ★ Prompt-assisted Modality Aligning
    ├── tests/test_mygo.py          # CPU smoke tests, no dataset required
    └── utils/                      # config, dataset, dataloader, metrics, logger
```

---

## ⚙️ Installation

```bash
git clone https://github.com/JashinKorone/MyGO.git
cd MyGO

# Python 3.8+ is recommended
conda create -n mygo python=3.8 -y && conda activate mygo
pip install -r requirements.txt
```

Verify the implementation without touching the dataset — the smoke tests run on CPU and cover all
seven modality combinations plus every ablation variant. Think of it as a **soundcheck** before the
live:

```bash
cd src
python -m tests.test_mygo
```

```
[ok] forward pass over all 7 modality combinations
[ok] backward pass, loss = 1.8578
[ok] PMA warm-up + loss regularizer, scale range = [1.000, 2.000]
[ok] zoned memory weights = 0.125, 0.250, 0.500, 0.500, 1.000
...
All smoke tests passed.
```

---

## 📊 Dataset: FakeSV+

FakeSV+ extends [FakeSV](https://github.com/ICTMCG/FakeSV) into a **modality-incomplete** benchmark:
**827 fake** + **1,827 real** news videos over **738 events**, where each sample may take any of the
`2^M − 1 = 7` modality combinations while **at least one modality is always available**.

The global missing rate follows

<div align="center">

η = 1 − ( Σᵢ mᵢ ) / ( L × M )  &nbsp;&nbsp;(Eq. 15)

</div>

with `mᵢ` the number of available modalities of sample `i`, `L` the sample count and `M = 3`.

### Realised combination distribution (%)

| Combination | η = 0 | η = 0.1 | η = 0.3 | η = 0.5 | η = 0.7 |
|:--|--:|--:|--:|--:|--:|
| Text | – | 0.94 | 6.88 | 16.17 | 34.11 |
| Audio | – | 0.80 | 6.86 | 15.92 | 33.42 |
| Video | – | 0.93 | 6.84 | 16.45 | 32.47 |
| Text + Audio | – | 8.62 | 14.63 | 13.12 | – |
| Video + Audio | – | 8.11 | 14.85 | 12.61 | – |
| Text + Video | – | 8.26 | 15.64 | 13.27 | – |
| Text + Audio + Video | 100 | 72.32 | 34.39 | 12.90 | – |

### Preparation

1. **Sign the agreement** in the [FakeSV repository](https://github.com/ICTMCG/FakeSV) to obtain the
   original data, and place the extracted frame-level features under `dataset/FakeSV/`:

   ```
   dataset/FakeSV/
   ├── embs/<video_id>.pkl        # {'vision': (K,1024), 'audio': (K,1024), 'text': (K,1024)}
   └── data/
       ├── data.json              # annotations, one JSON object per line
       ├── event/{1..5}/{train,test}.txt     # Event-split (E.S.)
       └── time/{train,val,test}.txt         # Temporal-split (T.S.)
   ```

   `embs/*.pkl` may optionally carry a `'caption'` entry with pre-extracted embedded-caption
   features; CKA falls back to the textual features when it is absent.

2. **Generate a FakeSV+ variant** for the missing rate you need:

   ```bash
   cd src
   python dataset_mask.py --dataset FakeSV --missing_rate 0.3 --seed 2024
   ```

   This writes `dataset/FakeSV-30/` containing zero-padded `embs/`, the per-video `masker.json`
   (`1` = missing) and a `statistics.json` reporting the realised distribution above.

3. **Two split schemes** are supported and selected in `src/configs/dataset/*.yaml`:

   | Scheme | `dataset_mode` | Protocol |
   |:--|:--|:--|
   | Event-split (E.S.) | `event` | 5-fold cross validation, 80 % train / 20 % test |
   | Temporal-split (T.S.) | `time` | chronological, earliest 70 % / next 15 % / latest 15 % |

---

## 🚀 Usage

### Train MyGO

```bash
cd src

# η = 0.3, event-split
python main.py --model MyGO --dataset FakeSV-30

# modality-complete setting (η = 0)
python main.py --model MyGO --dataset FakeSV

# override paper hyper-parameters on the fly
python main.py --model MyGO --dataset FakeSV-50 \
               --learning_rate 5e-5 --batch_size 256 \
               --prompt_top_k 10 --warmup_epochs 5
```

### Reproduce the ablations (Table 4)

<sub>♪ Every ablation is a member skipping rehearsal — the setlist still plays, just not as well.</sub>

```bash
python main.py --model MyGO --dataset FakeSV-30 --use_caption_weight False --use_prompt False  # -w/o CKA
python main.py --model MyGO --dataset FakeSV-30 --disentangle False                            # -w/o MDN
python main.py --model MyGO --dataset FakeSV-30 --use_pma False                                 # -w/o PMA
python main.py --model MyGO --dataset FakeSV-30 --use_pma_loss False                           # -w/o PMA loss
python main.py --model MyGO --dataset FakeSV-30 --ctrs_loss_wgt 0                               # -w/o ctrs loss
python main.py --model MyGO --dataset FakeSV-30 --orth_loss_wgt 0                               # -w/o orth loss
```

### Plug PMA into a baseline (Table 6)

`PMA` only rescales an existing *un-reduced* per-sample loss, so it can **join any band as a session
drummer** in three lines:

```python
from models.modules import PromptAssistedModalityAligning

self.pma = PromptAssistedModalityAligning(config)          # in __init__
loss_per_sample = F.cross_entropy(logits, label, reduction='none')
loss, _ = self.pma(prompt, loss_per_sample)                # in calculate_loss
# and call self.pma.step_epoch() from post_epoch_processing()
```

```bash
python main.py --model SVFEND --dataset FakeSV-30 --use_pma True     # SVFEND + PMA
python main.py --model SVFEND --dataset FakeSV-30 --use_pma False    # vanilla SVFEND
```

### Key hyper-parameters

| Argument | Config key | Default | Paper reference |
|:--|:--|--:|:--|
| `--batch_size` | `batch_size` | 256 | Sec. 4.3, co-tuned with `K` |
| `--learning_rate` | `learning_rate` | 1e-4 | searched in `[1e-4, 5e-5, 1e-3]` |
| `--ctrs_loss_wgt` | `ctrs_loss_wgt` | 0.3 | α of Eq. 14 |
| `--orth_loss_wgt` | `orth_loss_wgt` | 0.2 | β of Eq. 14 |
| `--cl_temp` | `cl_temp` | 0.2 | τ of Eq. 6 |
| `--prompt_top_k` | `prompt_top_k` | 10 | *K* prompt candidates per batch |
| `--warmup_epochs` | `warmup_epochs` | 5 | warm-up stage of Fig. 5 |
| — | `pma_active_window` | 5 | active-memory sliding window |
| — | `pma_decay` | 0.9 | inactive-memory decay |
| — | `fea_dim` / `dropout` / `num_heads` | 128 / 0.1 / 4 | Sec. 4.3 |

Training logs land in `src/log/`, checkpoints (when `save_model: True`) in `src/checkpoints/`.
Besides overall Accuracy / F1, the trainer prints a **per modality-combination** breakdown that
mirrors Table 5 of the paper:

```
modality-wise test result:
  0000  Video+Text+Audio   n=182   acc=0.7653  f1=0.7641
  1001  Video+Text         n=88    acc=0.7496  f1=0.7480
  1100  Text+Audio         n=79    acc=0.7260  f1=0.7248
  1101  Text               n=41    acc=0.6819  f1=0.6803
  ...
```

The PMA memory is logged after every epoch as well, which makes the discovery of weak modality
combinations directly observable:

```
PMA memory (epoch 12, 7 prompts) top weak combinations -> 1101:56.0, 1011:38.4, 1100:25.1, ...
```

---

## 📈 Results

### Main comparison — accuracy (%) under Event-split

| Model | η = 0 | η = 0.1 | η = 0.3 | η = 0.5 | η = 0.7 | Average |
|:--|--:|--:|--:|--:|--:|--:|
| TikTec | 73.89 | 72.23 | 68.23 | 60.35 | 57.32 | 66.40 |
| SVFEND | 76.15 | 74.05 | 70.75 | 66.15 | 63.65 | 70.27 |
| FakingRecipe | 77.91 | 74.77 | 69.12 | 64.34 | 61.30 | 69.48 |
| TATE | 75.87 | 73.59 | 70.52 | 67.57 | 63.33 | 70.16 |
| COM | 73.94 | 72.25 | 70.69 | 67.69 | 64.01 | 69.71 |
| **MyGO** | **79.05** | **76.11** | **74.01** | **72.10** | **70.04** | **74.06** |
| **Δ SOTA** | ↑1.14 | ↑1.34 | ↑3.26 | ↑4.53 | **↑6.03** | **↑3.79** |

Under Temporal-split MyGO reaches **77.10 %** average accuracy, i.e. **+4.85 %** over the best
baseline. The gap grows monotonically with the missing rate — exactly the robustness the model is
designed for.

> <sub>♪ Note the shape of the curve: baselines fall apart the moment a member is missing, while MyGO
> keeps the song going. At η = 0.7, where **every** sample has lost two of its three modalities,
> MyGO still holds 70.04 % — a **+6.03 %** lead. 迷子でも、前へ進める。</sub>

### Module-wise ablation — accuracy (%), Event-split

| Variant | η = 0 | η = 0.1 | η = 0.3 | η = 0.5 | η = 0.7 |
|:--|--:|--:|--:|--:|--:|
| **MyGO** | **79.05** | **76.11** | **74.01** | **72.10** | **70.04** |
| − w/o CKA | 77.78 | 75.21 | 73.51 | 71.89 | 69.83 |
| − w/o MDN | 78.31 | 74.64 | 70.28 | 68.33 | 67.21 |
| − w/o PMA | 78.92 | 75.63 | 72.19 | 68.98 | 66.14 |
| − w/o ctrs loss | 78.33 | 75.75 | 73.51 | 69.66 | 65.78 |
| − w/o orth loss | 77.91 | 74.33 | 72.11 | 70.88 | 68.26 |
| − w/o PMA loss | 79.04 | 75.92 | 73.00 | 69.91 | 67.31 |

**Takeaways.** MDN is the single most important module (−3.73 % at η = 0.3); PMA and the event-level
contrastive loss become critical precisely when data is severely incomplete (−3.90 % and −4.26 % at
η = 0.7).

### PMA as a plug-in — accuracy gain (%)

| Backbone | η = 0.1 | η = 0.3 | η = 0.5 | η = 0.7 |
|:--|--:|--:|--:|--:|
| TKCM + PMA | ↑0.04 | ↑1.44 | ↑2.06 | ↑3.11 |
| TwtrDetective + PMA | ↑0.43 | ↑0.37 | ↑1.40 | ↑2.49 |
| SVFEND + PMA | ↑0.50 | ↑1.14 | ↑0.96 | **↑3.33** |

---

## 💻 Environment

Experiments in the paper were run on a Linux server (CentOS 7) with Intel® Xeon® Gold 5218 CPUs,
an NVIDIA Tesla V100 PCIe 32 GB and 256 GB RAM, using PyTorch and HuggingFace Transformers.
The code also runs on CPU for debugging (see the smoke tests).

---

## 🔭 Future Work

- **ASR-assisted reconstruction** of missing text from audio, with noise-robust fusion tailored to
  the messy acoustics of short-video platforms.
- **Temporal modelling of evolving events**, e.g. temporal graph networks, to capture how
  misinformation propagates over time.

<sub>♪ In other words: the band is not disbanding. 次のライブでまた会いましょう。</sub>

---

## 📝 Citation

If this repository helps your research, please cite:

```bibtex
@article{Qiu2026,
  author    = {Qiu, Mingjie and Tan, Zhiyi and Bao, Bing-Kun},
  title     = {MyGO: Modality-incomplete Fake News Video Detection via Prompt-assisted Modality Disentangling Model},
  journal   = {ACM Trans. Multimedia Comput. Commun. Appl.},
  volume    = {22},
  number    = {2},
  year      = {2026},
  issue_date = {February 2026},
  issn      = {1551-6857},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3785481},
  url       = {https://doi.org/10.1145/3785481}
}
```

---

## 🙏 Acknowledgment

This work was supported by the National Natural Science Foundation of China (No. 62325206,
62532003), the Key R&D Program of Jiangsu Province (BE2023016-4) and the Natural Science Foundation
of Jiangsu Province (BK20210595).

We thank the authors of [FakeSV](https://github.com/ICTMCG/FakeSV) for releasing the base dataset.

And, of course, a nod to **MyGO!!!!!** — for lending us both a name and a reminder that being lost is
not the end of the story.

## 📄 License

Released under the MIT License for academic use. The FakeSV data is governed by its own agreement —
please follow the original terms.

*BanG Dream! It's MyGO!!!!!* and its characters are the property of Bushiroad / Craft Egg / Sanzigen.
This repository is an academic project with no affiliation to, or endorsement by, the rights holders;
the references above are affectionate homage only.

---

<div align="center">

<i>「私たちは、まだ迷子のままだけど。」</i><br/>
<sub>Even a model that has lost a modality can still find its way. That is the whole point of MyGO.</sub>

<br/><br/>

<sub>Questions or issues? Open an <a href="https://github.com/JashinKorone/MyGO/issues">issue</a> or contact <a href="mailto:2023010212@njupt.edu.cn">2023010212@njupt.edu.cn</a>.</sub>

</div>
