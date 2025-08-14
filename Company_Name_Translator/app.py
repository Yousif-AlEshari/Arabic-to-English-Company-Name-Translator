# streamlit_app.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st
import torch

# Import from your package (adjust if you use package prefix)
from io_artifacts import load_artifacts
from routing import translate_smart, is_likely_name
from org_renderer import render_org_name_en, normalize_tokens_ar
from config import ARTIFACTS_DIR

st.set_page_config(page_title="AR→EN Company Name Translator", layout="wide")

st.title("🧪 Arabic → English Org Name Translator (Hybrid)")
st.write("Live test the saved model + org-name renderer. Works fully offline once artifacts are saved.")

# -----------------------------
# Inline settings (no sidebar)
# -----------------------------
artifacts_dir = st.text_input("Artifacts directory", ARTIFACTS_DIR)
device_opt = st.radio("Device", ["Auto (CUDA if available)", "CPU"], horizontal=True)
device = torch.device("cuda" if (device_opt.startswith("Auto") and torch.cuda.is_available()) else "cpu")

@st.cache_resource(show_spinner=True)
def _load(art_dir: str, device_type: str):
    if not os.path.exists(os.path.join(art_dir, "model.pt")):
        raise FileNotFoundError(
            f"No model.pt in '{art_dir}'. Train once using your training script to create: "
            "model.pt, src_vocab.json, trg_vocab.json, config.json."
        )
    model, src_vocab, trg_vocab, cfg = load_artifacts(art_dir, torch.device(device_type))
    return model, src_vocab, trg_vocab, cfg

try:
    model, src_vocab, trg_vocab, cfg = _load(artifacts_dir, device.type)
    st.success(f"Loaded ✓  (AR vocab: {len(src_vocab)} | EN vocab: {len(trg_vocab)} | device: {device.type})")
    with st.expander("Model config", expanded=False):
        st.json({
            "encoder_embedding_size": cfg.encoder_embedding_size,
            "decoder_embedding_size": cfg.decoder_embedding_size,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "enc_dropout": cfg.enc_dropout,
            "dec_dropout": cfg.dec_dropout,
        })
except Exception as e:
    st.error(str(e))
    st.stop()

# -----------------------------
# Single input tester
# -----------------------------
st.subheader("🔤 Single name/sentence")

default_text = ""
text = st.text_input("Arabic input", default_text)

colA, colB = st.columns([1, 1])
with colA:
    run_btn = st.button("Translate", use_container_width=True)
with colB:
    show_debug = st.toggle("Show debug (routing & tokenization)")

def translate_once(txt: str) -> str:
    return translate_smart(model, txt, src_vocab, trg_vocab, device)

if run_btn or text:
    try:
        output = translate_once(text)
        st.markdown("**Output (EN):**")
        st.success(output)

        if show_debug:
            st.divider()
            st.markdown("**Debug**")
            route = "Org renderer" if is_likely_name(text) else "Seq2Seq model"
            st.write(f"Routing: **{route}**")
            toks_norm = normalize_tokens_ar(text)
            st.write("Normalized Arabic tokens:", toks_norm)
            if route == "Org renderer":
                st.write("Renderer preview:", render_org_name_en(text))
    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Batch mode (CSV/XLSX)
# -----------------------------
st.subheader("📄 Batch translate file")
st.caption("Upload a CSV or Excel. Pick the column with Arabic names. Get a downloadable CSV back.")

uploaded = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"])
col_name = st.text_input("Column name that contains Arabic text", "Original_Arabic_Name")
run_batch = st.button("Run batch", type="primary")

@st.cache_data(show_spinner=True)
def _batch_translate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    def _safe_translate(x):
        try:
            return translate_once(str(x))
        except Exception:
            return ""
    df["Translated_Output_Streamlit"] = df[col].apply(_safe_translate)
    return df

if run_batch and uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        else:
            df_in = pd.read_excel(uploaded)
        if col_name not in df_in.columns:
            st.error(f"Column '{col_name}' not in file. Found columns: {list(df_in.columns)}")
        else:
            df_out = _batch_translate(df_in, col_name)
            st.success(f"Translated {len(df_out)} rows.")
            st.dataframe(df_out, use_container_width=True)
            csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="translated_output.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Batch error: {e}")
elif run_batch and uploaded is None:
    st.warning("Please upload a file first.")
