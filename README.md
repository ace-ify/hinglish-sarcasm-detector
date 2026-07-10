# 🎭 Hinglish Sarcasm Detector

Detects sarcasm in **Hinglish** (Hindi–English code-mixed) text using a fine-tuned
[MuRIL](https://huggingface.co/google/muril-base-cased) transformer.

**Live app:** _(add your Streamlit URL after deploy)_
**Model:** [naimish2169/hinglish-sarcasm-detector](https://huggingface.co/naimish2169/hinglish-sarcasm-detector)

## Results

Fine-tuned MuRIL, evaluated on a held-out validation set (1,935 samples):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| not sarcasm | 0.98 | 0.97 | 0.97 |
| sarcasm | 0.95 | 0.97 | 0.96 |
| **accuracy** | | | **0.97** |

Decision threshold calibrated to **0.44** (max macro-F1 on validation).

## How it was built

1. **Data** — Hinglish sarcasm dataset (binary labels). Its "not sarcasm" class was
   mostly news/political tweets, so the model over-flagged normal conversation. Fixed by
   mixing in genuine conversational Hinglish (neutral/joy/love/admiration) from an
   emotion dataset as extra *not-sarcasm* examples.
2. **Model** — `google/muril-base-cased` (pretrained on Indian languages + their Roman
   transliteration) fine-tuned for binary classification, `max_length=128`.
3. **Calibration** — decision threshold picked from the validation set, not hard-coded 0.5.

Training notebook: `Hinglish_Sarcasm_Train.ipynb`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

Push this repo to GitHub, then go to [share.streamlit.io](https://share.streamlit.io),
pick the repo + `app.py`, and deploy. The model is pulled from Hugging Face at runtime,
so it is **not** stored in this repo.
