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
    st.write("""
    Analyzing the Fake News Dataset:
    - Class distribution
    - Text length analysis
    - Content visualization
    """)

    @st.cache_data(show_spinner="Loading dataset...")
    def load_data():
        try:
            # Método 1: Cargar directamente con datasets
            try:
                # Especificamos exactamente el split que queremos
                dataset = load_dataset(
                    "ErfanMoosaviMonazzah/fake-news-detection-dataset-English",
                    split="train",  # Especificamos el split directamente
                    verification_mode="no_checks"  # Evita verificaciones que pueden fallar
                )
                df = dataset.to_pandas()
                
                # Verificación básica de columnas
                if "text" not in df.columns:
                    if "article" in df.columns:
                        df["text"] = df["article"]
                    elif "content" in df.columns:
                        df["text"] = df["content"]
                
                if "label" not in df.columns:
                    if "is_fake" in df.columns:
                        df["label"] = df["is_fake"]
                    elif "target" in df.columns:
                        df["label"] = df["target"]
                
                return df[["text", "label"]].dropna()
            
            except Exception as e:
                st.warning(f"Primary load failed: {str(e)}")
                raise
                
        except Exception:
            # Método 2: Datos de ejemplo embebidos
            sample_data = {
                "text": [
                    "Scientific study confirms benefits of exercise (Real)",
                    "Aliens visiting Earth, government confirms (Fake)",
                    "New economic policy announced (Real)",
                    "Celebrity reveals shocking secret (Fake)"
                ],
                "label": [1, 0, 1, 0]
            }
            return pd.DataFrame(sample_data)

    # Cargar y procesar datos
    try:
        df = load_data()
        
        # Conversión robusta de labels
        df["label"] = df["label"].apply(
            lambda x: 0 if str(x).lower() in ["fake", "false", "0", "no"] else 1
        )
        
        # Visualizaciones
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Class Distribution")
            fig1 = px.pie(
                df,
                names=df["label"].map({0: "Fake", 1: "Real"}),
                color_discrete_sequence=["#FF5733", "#33FF57"]
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Text Length Distribution")
            df["length"] = df["text"].str.split().str.len()
            fig2 = px.histogram(
                df,
                x="length",
                color="label",
                color_discrete_map={0: "#FF5733", 1: "#33FF57"},
                nbins=30
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Word Cloud opcional
        if st.checkbox("Show Word Clouds"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Fake News Word Cloud")
                fake_text = " ".join(df[df["label"]==0]["text"].astype(str))
                if fake_text.strip():
                    wc = WordCloud(width=600, height=300).generate(fake_text)
                    plt.imshow(wc)
                    plt.axis("off")
                    st.pyplot(plt)
            
            with col2:
                st.write("Real News Word Cloud")
                real_text = " ".join(df[df["label"]==1]["text"].astype(str))
                if real_text.strip():
                    wc = WordCloud(width=600, height=300).generate(real_text)
                    plt.imshow(wc)
                    plt.axis("off")
                    st.pyplot(plt)
    
    except Exception as e:
        st.error(f"Failed to process data: {str(e)}")

elif page == "3️⃣ Hyperparam Tuning":
    st.title("⚙️ Hyperparameter Optimization")
    st.markdown("""
    ## Fine-tuning de Modelos
    Optimización de hiperparámetros para los modelos:
    - DistilBERT
    - BERT
    - T5
    """)
    
    model_type = st.selectbox("Selecciona el modelo a optimizar", 
                            ["DistilBERT", "BERT", "T5"])
    
    # Configuración común
    st.subheader("🔧 Configuración del Estudio")
    n_trials = st.slider("Número de trials", 10, 100, 20)
    metric = st.selectbox("Métrica a optimizar", 
                         ["f1", "accuracy", "precision", "recall"])
    
    if st.button("🚀 Ejecutar Optimización"):
        with st.spinner(f"Optimizando {model_type} (esto puede tomar varios minutos)..."):
            best_params = run_optuna_study(model_type, n_trials, metric)
            
            st.success("¡Optimización completada!")
            st.subheader("🎯 Mejores Parámetros Encontrados")
            st.json(best_params)
            
            # Visualización de resultados
            display_results(model_type)

def run_optuna_study(model_type, n_trials, metric):
    import optuna
    from transformers import Trainer, TrainingArguments
    
    def objective(trial):
        # Parámetros comunes
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "num_train_epochs": trial.suggest_int("epochs", 1, 5),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }
        
        # Parámetros específicos por modelo
        if model_type == "DistilBERT":
            params.update({
                "hidden_dropout_prob": trial.suggest_float("hidden_dropout", 0.1, 0.5),
            })
        elif model_type == "BERT":
            params.update({
                "attention_probs_dropout_prob": trial.suggest_float("attention_dropout", 0.1, 0.3),
            })
        elif model_type == "T5":
            params.update({
                "dropout_rate": trial.suggest_float("dropout", 0.1, 0.4),
            })
        
        # Configuración de entrenamiento
        training_args = TrainingArguments(
            output_dir=f"./results_{model_type}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=metric,
            **params
        )
        
        # Cargar modelo y datasets (adaptar según tu implementación)
        model = load_your_model(model_type)  # Reemplaza con tu función
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics  # Asegúrate de definir esta función
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        return eval_results[f"eval_{metric}"]
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def display_results(model_type):
    # Gráficos de optimización
    st.subheader("📈 Resultados de la Optimización")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(
            x=range(1, n_trials+1),
            y=study.best_trials,
            title="Evolución de la Métrica",
            labels={"x": "Trial", "y": metric}
        ), use_container_width=True)
    
    with col2:
        st.plotly_chart(px.parallel_coordinates(
            study.trials_dataframe(),
            color=f"eval_{metric}",
            title="Combinación de Parámetros"
        ), use_container_width=True)
    
    # Exportar resultados
    if st.button("💾 Exportar Configuración Óptima"):
        save_best_config(model_type, study.best_params)
