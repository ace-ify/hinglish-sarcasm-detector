import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

MODEL_PATH = "naimish2169/hinglish-sarcasm-detector"   # Hugging Face repo
MAX_LEN = 128
THRESHOLD = 0.44  

st.set_page_config(page_title="Hinglish Sarcasm Detector", page_icon="🎭")


def normalize_hinglish(text):
    text = str(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)     # brooo -> broo
    text = re.sub(r'([!?.]){2,}', r'\1\1', text)   # !!!! -> !!
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    mdl.eval()
    return tok, mdl


def predict(text, tok, mdl):
    enc = tok(normalize_hinglish(text), return_tensors="pt",
              truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        prob = torch.softmax(mdl(**enc).logits, -1)[0, 1].item()
    return prob


st.title("🎭 Hinglish Sarcasm Detector")
st.caption("MuRIL fine-tuned for code-mixed (Hindi + English) sarcasm detection")

tok, mdl = load_model()

text = st.text_area("Enter a sentence (Hinglish / English):",
                    placeholder="Wah! 2 ghante late ho aur bol rahe ho 'bas 5 min'. Genius!")

col1, col2 = st.columns([1, 3])
detect = col1.button("Detect", type="primary")

if detect and text.strip():
    prob = predict(text, tok, mdl)
    pct = prob * 100
    st.progress(prob)
    if prob >= THRESHOLD:
        st.error(f"🌀 **SARCASM**  ·  {pct:.1f}% confidence")
    else:
        st.success(f"🙂 **Not sarcasm**  ·  {100 - pct:.1f}% confidence")

st.divider()
st.subheader("Try an example")
examples = [
    "Bhai kya hi logic lagaya hai tune, Nobel prize milna chahiye! /s",
    "Bohot badhiya service hai aapki, 10 din se call wait pe hai. Amazing.",
    "Love this product, sach mein badiya kaam kiya team ne!",
    "Mast weather hai yaar, chalo chai peete hain!",
]
for ex in examples:
    if st.button(ex, key=ex):
        prob = predict(ex, tok, mdl)
        pct = prob * 100
        label = "🌀 SARCASM" if prob >= THRESHOLD else "🙂 Not sarcasm"
        st.write(f"**{label}** — {pct:.1f}%  ·  _{ex}_")
