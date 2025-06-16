import streamlit as st
import torch
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BertTokenizerFast,
    BertForSequenceClassification
)
# ——————————————————————————————————————
# ¡Ésta tiene que ser la PRIMERA llamada a Streamlit!
st.set_page_config(page_title="📰 Fake News Detection", layout="wide")
# ——————————————————————————————————————
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Usando dispositivo: {device}")
st.title("📰 Fake News Detection")
def load_models():
    models = {}
    # — T5 seq2seq (Fake/Real)
    t5_repo = "Alvaropad1/Fakenews"
    t5_subf = "t5-Fakenews"
    t5_tok  = T5TokenizerFast.from_pretrained(t5_repo, subfolder=t5_subf)
    t5_mod  = T5ForConditionalGeneration.from_pretrained(t5_repo, subfolder=t5_subf).to(device)
    fake_id = t5_tok.convert_tokens_to_ids("fake")
    real_id = t5_tok.convert_tokens_to_ids("real")
    models["T5"] = (t5_tok, t5_mod, fake_id, real_id)
#
    # — DistilBERT classifier
    db_repo = "Alvaropad1/Fakenews"
    db_subf = "Distilbert-fakenews"
    db_tok  = DistilBertTokenizerFast.from_pretrained(db_repo, subfolder=db_subf)
    db_mod  = DistilBertForSequenceClassification.from_pretrained(db_repo, subfolder=db_subf).to(device)
    models["DistilBERT"] = (db_tok, db_mod)
#
    # — BERT classifier
    bert_repo = "Alvaropad1/Fakenews"
    bert_subf = "bert-fakenews"
    # usamos vocab oficial de bert-base
    bert_tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert_mod  = BertForSequenceClassification.from_pretrained(bert_repo, subfolder=bert_subf).to(device)
    models["BERT"] = (bert_tok, bert_mod)
#
    return models
#
models = load_models()
#
# 3) Funciones de inferencia
def predict_t5(text: str):
    tok, mod, fake_id, real_id = models["T5"]
    inputs = tok("classify: " + text,
                 return_tensors="pt",
                 truncation=True,
                 max_length=256).to(device)
    with torch.no_grad():
        logits = mod(**inputs).logits  # [1, seq_len, vocab_size]
    probs    = torch.softmax(logits[:, 0, :], dim=-1)[0]
    p_fake   = probs[fake_id].item()
    p_real   = probs[real_id].item()
    label    = "fake" if p_fake > p_real else "real"
    return label, {"fake": p_fake, "real": p_real}
#
def predict_cls(text: str, model_key: str):
    tok, mod = models[model_key]
    inputs = tok(text,
                 return_tensors="pt",
                 truncation=True,
                 padding="max_length",
                 max_length=256).to(device)
    with torch.no_grad():
        logits = mod(**inputs).logits  # [1, num_labels]
    probs = torch.softmax(logits[0], dim=-1)
    id2lab = mod.config.id2label
    conf   = {id2lab[i]: probs[i].item() for i in range(len(probs))}
    label  = max(conf, key=conf.get)
    return label, conf
#
# 4) Streamlit UI
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection")
#
st.markdown(
    """
    Ingresa un texto de noticia y selecciona el modelo con el cual quieres clasificarlo como **real** o **fake** junto con sus puntajes de confianza.
    """
)
#
model_choice = st.selectbox("Selecciona modelo", ["T5", "DistilBERT", "BERT"])
text_input   = st.text_area("Texto de noticia:", height=200)
#
if st.button("🔍 Predecir"):
    if not text_input.strip():
        st.warning("Por favor ingresa algún texto antes de predecir.")
    else:
        with st.spinner(f"Analizando con {model_choice}..."):
            if model_choice == "T5":
                label, conf = predict_t5(text_input)
            else:
                label, conf = predict_cls(text_input, model_choice)
        st.subheader(f"**Predicción:** {label}")
        st.write("**Confianza:**")
        st.json(conf)
