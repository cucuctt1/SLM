# Run This Project on Google Colab (No .ipynb Needed)

You can run everything by opening a blank Colab notebook and pasting the cells below.

---

## 1) Open Colab and enable GPU

- Open: https://colab.research.google.com
- Create **New Notebook**
- Go to **Runtime -> Change runtime type -> T4 GPU**

---

## 2) Clone project into Colab

```bash
!git clone <YOUR_REPO_URL> /content/SLM
%cd /content/SLM
```

If your files are local only (not on GitHub), upload the folder manually to `/content/SLM`.

---

## 3) Install only allowed dependencies (no torch reinstall)

```bash
!bash colab_setup.sh
```

This installs: `kaggle`, `tqdm`, `numpy`.

---

## 4) Set Kaggle credentials

Create a Kaggle API key at: https://www.kaggle.com/settings (download `kaggle.json`).

Option A (recommended in Colab):

```python
import os
os.environ["KAGGLE_USERNAME"] = "YOUR_USERNAME"
os.environ["KAGGLE_KEY"] = "YOUR_KEY"
```

Option B:

```bash
!mkdir -p /root/.kaggle
# upload kaggle.json from your machine
from google.colab import files
files.upload()
!mv kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
```

---

## 5) (Optional) adjust training config

Edit `config.py` values if needed:
- `epochs`
- `max_steps` (set to a small value for quick testing)
- `dataset_slug`

Quick edit from Colab:

```bash
!sed -n '1,220p' /content/SLM/config.py
```

---

## 6) Start training

```bash
!python train.py
```

What this does automatically:
- Downloads + extracts Kaggle dataset
- Builds English corpus
- Trains manual byte-level BPE
- Runs mandatory sanity checks
- Trains GPT decoder-only model
- Saves checkpoints in `/content/checkpoints/`
- Saves quantized int4 model to `/content/checkpoints/quantized_int4.pt`

---

## 7) Run generation (full or quantized)

```bash
!python generate.py
```

`generate.py` prefers quantized checkpoint if available.

---

## 8) Resume interrupted training

Just run again:

```bash
!python train.py
```

It auto-resumes latest checkpoint from `/content/checkpoints/`.

---

## 9) Save artifacts to Google Drive (recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!mkdir -p /content/drive/MyDrive/SLM_outputs
!cp -r /content/checkpoints /content/drive/MyDrive/SLM_outputs/
!cp -r /content/tokenizer /content/drive/MyDrive/SLM_outputs/
```

---

## 10) Common issues

1. **Kaggle auth error**
   - Recheck `KAGGLE_USERNAME` and `KAGGLE_KEY`.

2. **Out of memory on T4**
   - Lower in `config.py`:
     - `batch_size` to `1`
     - increase `grad_accum_steps`
     - optionally lower `context_length`

3. **Slow first run**
   - Tokenizer training and dataset preprocessing are one-time heavy steps.

---

## Minimal quick-test run

If you only want to verify pipeline quickly, set in `config.py`:
- `epochs = 1`
- `max_steps = 20`
- `max_corpus_chars = 2_000_000`

Then run:

```bash
!python train.py
!python generate.py
```
