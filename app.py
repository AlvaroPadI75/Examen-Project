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
#-------------------------------------
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from huggingface_hub import hf_hub_download
# ——————————————————————————————————————
# ¡Ésta tiene que ser la PRIMERA llamada a Streamlit!
st.set_page_config(page_title="📰 Fake News Detection", layout="wide")
# ——————————————————————————————————————
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Usando dispositivo: {device}")
st.title("📰 Fake News Detection")
@st.cache_resource(show_spinner=False)
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
    # 1) Desempaqueta tokenizer + modelo + ids
    tok, mod, fake_id, real_id = models["T5"]

    # 2) Tokenización y envío a device
    inputs = tok(
        "classify fake or real news: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    generated = mod.generate(
        input_ids      = inputs.input_ids,
        attention_mask = inputs.attention_mask,
        max_new_tokens = 2,
        num_beams      = 5,
        early_stopping = True,
    )

    pred = tok.decode(generated[0], skip_special_tokens=True).strip().lower()
    conf = {"real": 0.0, "fake": 0.0}
    if pred in conf:
        conf[pred] = 1.0

    return pred, conf

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
# ————————————— SIDEBAR: NAVEGACIÓN —————————————
st.sidebar.title("Navegación")
page = st.sidebar.radio("", ["1️⃣ Inference", "2️⃣ Dataset EDA", "3️⃣ Hyperparam Tuning", "4️⃣ Model Analysis"])

# ————————————— PÁGINA 1: INTERFAZ DE INFERENCIA —————————————
if page == "1️⃣ Inference":
    st.title("📰 Fake News Detection – Inference")
    st.markdown(
        "Ingresa un texto de noticia y selecciona el modelo en la barra lateral."
    )
    model_choice = st.sidebar.selectbox("Modelo", ["T5", "DistilBERT", "BERT"])
    text_input   = st.text_area("🖋️ Texto de noticia:", height=200)

    if st.button("🔍 Predecir"):
        if not text_input.strip():
            st.warning("Por favor ingresa algún texto antes de predecir.")
        else:
            with st.spinner(f"Analizando con {model_choice}..."):
                if model_choice == "T5":
                    label, conf = predict_t5(text_input)
                else:
                    label, conf = predict_cls(text_input, model_choice)
            st.subheader(f"**Predicción:** {label.upper()}")
            st.write("**Confianzas:**")
            st.json(conf)
            
elif page == "2️⃣ Dataset EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write(
        """
        Here we explore the structure and content of our Fake-News dataset:
        - **Class distribution**  
        - **Token-length histogram**  
        - **Word clouds**  
        - **Sample noisy / ambiguous texts**
        """
    )

elif page == "2️⃣ Dataset EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write(
        """
        Here we explore the structure and content of our Fake-News dataset:
        - **Class distribution**  
        - **Token-length histogram**  
        - **Word clouds**  
        - **Sample noisy / ambiguous texts**
        """
    )

    @st.cache_data(show_spinner=False)
    def load_data():
        try:
            # Intento 1: Descargar el CSV directamente
            try:
                path = hf_hub_download(
                    repo_id="ErfanMoosaviMonazzah/fake-news-detection-dataset-English",
                    filename="train.csv",
                    repo_type="dataset"  # ¡Esto es crucial!
                )
                df = pd.read_csv(path)
                return df
            except:
                # Intento 2: Cargar a través de la API de datasets
                dataset = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")
                return dataset["train"].to_pandas()
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            # Datos de ejemplo como fallback
            return pd.DataFrame({
                "text": [
                    "Scientists confirm climate change is accelerating",
                    "Aliens visiting Earth, government confirms",
                    "New economic policy announced by president",
                    "Celebrity couple announces surprise wedding"
                ],
                "label": [1, 0, 1, 0]  # 1=Real, 0=Fake
            })

    # Cargar datos (correctamente indentado dentro del elif)
    df = load_data()

    # Verificar y preparar los datos
    if "label" not in df.columns or "text" not in df.columns:
        st.error("El dataset no tiene las columnas requeridas ('text' y 'label')")
        st.stop()

    # Convertir labels si son strings
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"fake": 0, "real": 1, "FAKE": 0, "REAL": 1})

    # 1) Distribución de clases
    st.subheader("📊 Class Distribution: Fake vs Real")
    fig1 = px.pie(
        df, 
        names=df["label"].map({0: "Fake", 1: "Real"}),
        hole=0.3,
        color_discrete_sequence=["#FF5733", "#33FF57"],
        title="Proportion of Fake vs Real News"
    )
    fig1.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Histograma de longitud de tokens
    st.subheader("📏 Token Count Distribution")
    df["token_count"] = df["text"].str.split().str.len()
    fig2 = px.histogram(
        df,
        x="token_count",
        nbins=50,
        color="label",
        color_discrete_map={0: "#FF5733", 1: "#33FF57"},
        labels={"token_count": "Number of Tokens", "label": "News Type"},
        category_orders={"label": [0, 1]},
        barmode="overlay",
        opacity=0.7,
        title="Distribution of Article Lengths (in Tokens)"
    )
    fig2.update_layout(legend_title_text="News Type")
    st.plotly_chart(fig2, use_container_width=True)

    # 3) Nubes de palabras
    st.subheader("🗯️ Word Clouds by News Type")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Fake News Word Cloud**")
        fake_text = " ".join(df[df["label"]==0]["text"].astype(str).tolist())
        wc_fake = WordCloud(
            width=500, 
            height=300, 
            background_color="white",
            colormap="Reds",
            stopwords=set(STOPWORDS)
        ).generate(fake_text)
        fig_fake, ax_fake = plt.subplots(figsize=(8, 5))
        ax_fake.imshow(wc_fake, interpolation="bilinear")
        ax_fake.axis("off")
        st.pyplot(fig_fake)
    
    with col2:
        st.markdown("**Real News Word Cloud**")
        real_text = " ".join(df[df["label"]==1]["text"].astype(str).tolist())
        wc_real = WordCloud(
            width=500, 
            height=300, 
            background_color="white",
            colormap="Greens",
            stopwords=set(STOPWORDS)
        ).generate(real_text)
        fig_real, ax_real = plt.subplots(figsize=(8, 5))
        ax_real.imshow(wc_real, interpolation="bilinear")
        ax_real.axis("off")
        st.pyplot(fig_real)

    # 4) Ejemplos de texto
    st.subheader("📝 Sample News Texts")
    tab1, tab2 = st.tabs(["Real News", "Fake News"])
    
    with tab1:
        st.markdown("**Examples of Real News:**")
        real_samples = df[df["label"]==1].sample(3, random_state=42)
        for idx, row in real_samples.iterrows():
            st.markdown(f"""
            <div style="background-color:#e6ffe6; padding:10px; border-radius:5px; margin:10px 0;">
            <b>Sample {idx}</b><br>
            {row['text'][:300]}...
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("**Examples of Fake News:**")
        fake_samples = df[df["label"]==0].sample(3, random_state=42)
        for idx, row in fake_samples.iterrows():
            st.markdown(f"""
            <div style="background-color:#ffe6e6; padding:10px; border-radius:5px; margin:10px 0;">
            <b>Sample {idx}</b><br>
            {row['text'][:300]}...
            </div>
            """, unsafe_allow_html=True)
