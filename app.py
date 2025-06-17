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
def objective(trial, model_type, metric):
    # 1. Definir parámetros a optimizar
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("epochs", 1, 5),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }
    
    # 2. Parámetros específicos por modelo
    if model_type == "DistilBERT":
        params["hidden_dropout_prob"] = trial.suggest_float("hidden_dropout", 0.1, 0.5)
    elif model_type == "BERT":
        params["attention_probs_dropout_prob"] = trial.suggest_float("attention_dropout", 0.1, 0.3)
    elif model_type == "T5":
        params["dropout_rate"] = trial.suggest_float("dropout", 0.1, 0.4)
    
    # 3. Configuración del entrenamiento (adaptar a tu implementación)
    training_args = TrainingArguments(
        output_dir=f"./results_{model_type}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        **params
    )
    
    # 4. Cargar modelo y datasets (implementar según tu caso)
    model = load_model(model_type)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # 5. Entrenamiento y evaluación
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results[f"eval_{metric}"]
    
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
    st.title("⚙️ Optimización de Hiperparámetros")
    st.markdown("""
    ## Configuración del Proceso de Optimización
    Ajuste fino de los parámetros del modelo para maximizar el rendimiento
    """)

    # 1. Selector de modelo y configuración
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox(
            "Modelo a optimizar",
            ["DistilBERT", "BERT", "T5"],
            help="Selecciona qué modelo deseas optimizar"
        )
    
    with col2:
        n_trials = st.slider(
            "Número de trials",
            min_value=5,
            max_value=50,
            value=15,
            help="Cantidad de experimentos a realizar"
        )
    
    with col3:
        metric = st.selectbox(
            "Métrica objetivo",
            ["f1", "accuracy", "precision", "recall"],
            index=0,
            help="Métrica principal a optimizar"
        )

    # 2. Panel de parámetros con explicaciones
    with st.expander("📋 Detalles de los Parámetros a Optimizar", expanded=True):
        st.markdown("""
        | Parámetro          | Rango               | Importancia               |
        |--------------------|---------------------|---------------------------|
        | Learning Rate      | 1e-6 a 1e-4         | Controla la velocidad de aprendizaje |
        | Batch Size         | 8, 16, 32           | Afecta memoria y estabilidad |
        | Epochs             | 1 a 5               | Evitar sobreentrenamiento  |
        | Weight Decay       | 0.0 a 0.3           | Regularización L2         |
        | Dropout            | Modelo-específico   | Prevenir overfitting      |
        """)

    # 3. Función principal de optimización
    if st.button("🚀 Ejecutar Optimización", type="primary"):
        with st.spinner(f"Optimizando {model_type} - Esto puede tomar varios minutos..."):
            try:
                # Configuración de Optuna
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(),
                    pruner=optuna.pruners.HyperbandPruner()
                )
                
                # Ejecutar optimización
                study.optimize(
                    lambda trial: objective(trial, model_type, metric),
                    n_trials=n_trials,
                    show_progress_bar=True
                )

                # Mostrar resultados
                st.success("Optimización completada!")
                
                # 4. Visualización de resultados
                st.subheader("📊 Resultados de la Optimización")
                
                # Gráfico de evolución
                fig1 = px.line(
                    study.trials_dataframe(),
                    x="number",
                    y="value",
                    title=f"Evolución de la métrica {metric}",
                    labels={"value": metric, "number": "Trial"}
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Mejores parámetros
                st.subheader("🎯 Mejores Parámetros Encontrados")
                best_params = study.best_params
                st.json(best_params)
                
                # Gráfico de importancia
                fig2 = optuna.visualization.plot_param_importances(study)
                st.pyplot(fig2)
                
                # 5. Explicación de resultados
                with st.expander("🔍 Interpretación de Resultados"):
                    st.markdown(f"""
                    - **Mejor {metric} obtenido**: {study.best_value:.4f}
                    - **Trial número**: {study.best_trial.number}
                    - **Parámetros clave**:
                        - Learning Rate: {best_params.get('learning_rate', 'N/A')}
                        - Batch Size: {best_params.get('per_device_train_batch_size', 'N/A')}
                    """)
                    
                    if model_type == "DistilBERT":
                        st.write("Para DistilBERT, el dropout óptimo fue:", best_params.get("hidden_dropout_prob", "N/A"))
                    elif model_type == "BERT":
                        st.write("Para BERT, el attention dropout óptimo fue:", best_params.get("attention_probs_dropout_prob", "N/A"))
                
                # 6. Opción para guardar resultados
                if st.button("💾 Guardar Configuración Óptima"):
                    save_best_config(model_type, best_params)
                    st.toast("Configuración guardada exitosamente!", icon="✅")
            
            except Exception as e:
                st.error(f"Error durante la optimización: {str(e)}")
                st.error("Revisa los logs para más detalles")

# Función de optimización (debe estar en el mismo archivo o importarse)
    
elif page == "4️⃣ Model Analysis":
    st.title("📊 Análisis de Modelos")
    st.markdown("""
    ## Evaluación Comparativa de Modelos
    Análisis detallado del rendimiento de cada arquitectura
    """)

    # 1. Configuración inicial
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        from transformers import pipeline
    except ImportError as e:
        st.error(f"Error de importación: {str(e)}")
        st.stop()

    # 2. Selección de modelo para análisis
    model_type = st.radio(
        "Modelo a analizar",
        ["DistilBERT", "BERT", "T5"],
        horizontal=True
    )

    # 3. Sección de métricas generales
    with st.container():
        st.subheader("📈 Rendimiento General")
        col1, col2, col3 = st.columns(3)
        
        # Datos de ejemplo (reemplazar con tus métricas reales)
        metrics = {
            "DistilBERT": {"accuracy": 0.89, "precision": 0.88, "recall": 0.90, "f1": 0.89},
            "BERT": {"accuracy": 0.91, "precision": 0.90, "recall": 0.92, "f1": 0.91},
            "T5": {"accuracy": 0.87, "precision": 0.86, "recall": 0.88, "f1": 0.87}
        }
        
        with col1:
            st.metric("Accuracy", f"{metrics[model_type]['accuracy']:.2%}")
            st.metric("Precision", f"{metrics[model_type]['precision']:.2%}")
        
        with col2:
            st.metric("Recall", f"{metrics[model_type]['recall']:.2%}")
            st.metric("F1-Score", f"{metrics[model_type]['f1']:.2%}")
        
        with col3:
            st.write("**Dataset:** FakeNewsNet")
            st.write("**Split:** Test (30% del total)")
            st.write(f"**Muestras:** 12,543 artículos")

    # 4. Matriz de confusión
    with st.expander("🧮 Matriz de Confusión", expanded=True):
        # Datos de ejemplo
        cm = np.array([[1250, 150], [120, 1280]]) if model_type == "DistilBERT" else \
             np.array([[1300, 100], [90, 1310]]) if model_type == "BERT" else \
             np.array([[1200, 200], [180, 1220]])
        
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fake', 'Real'], 
                    yticklabels=['Fake', 'Real'])
        plt.ylabel('Verdaderos')
        plt.xlabel('Predicciones')
        plt.title(f'Matriz de Confusión - {model_type}')
        st.pyplot(fig)

    # 5. Reporte de clasificación
    with st.expander("📝 Reporte de Clasificación Detallado"):
        # Datos de ejemplo
        st.code(f"""
        Classification Report - {model_type}
        {'-'*50}
        precision    recall  f1-score   support

        Fake          {metrics[model_type]['precision']:.2f}     {metrics[model_type]['recall']-0.02:.2f}     {metrics[model_type]['f1']-0.01:.2f}      6234
        Real          {metrics[model_type]['precision']+0.01:.2f}     {metrics[model_type]['recall']:.2f}     {metrics[model_type]['f1']+0.01:.2f}      6309

        accuracy                            {metrics[model_type]['accuracy']:.2f}     12543
        macro avg     {metrics[model_type]['precision']+0.005:.2f}     {metrics[model_type]['recall']-0.01:.2f}     {metrics[model_type]['f1']:.2f}     12543
        weighted avg  {metrics[model_type]['precision']:.2f}     {metrics[model_type]['recall']:.2f}     {metrics[model_type]['f1']:.2f}     12543
        """)

    # 6. Análisis de errores
    with st.expander("🔍 Análisis de Errores"):
        st.subheader("Ejemplos de Falsos Positivos/Negativos")
        
        # Datos de ejemplo
        error_samples = {
            "DistilBERT": [
                {"text": "El presidente anuncia nueva ley de impuestos...", "true": "Real", "pred": "Fake", "reason": "Vocabulario político"},
                {"text": "Descubren cura milagrosa para el cáncer...", "true": "Fake", "pred": "Real", "reason": "Lenguaje científico mal utilizado"}
            ],
            "BERT": [
                {"text": "Terremoto de 8.5 grados golpea la costa...", "true": "Real", "pred": "Fake", "reason": "Evento extremo"},
                {"text": "Celebridad revela que es un reptiliano...", "true": "Fake", "pred": "Real", "reason": "Sensacionalismo"}
            ],
            "T5": [
                {"text": "Nuevo estudio sobre cambio climático...", "true": "Real", "pred": "Fake", "reason": "Términos técnicos"},
                {"text": "Vacuna causa autismo, según médico...", "true": "Fake", "pred": "Real", "reason": "Pseudociencia"}
            ]
        }
        
        for error in error_samples[model_type]:
            with st.container(border=True):
                st.markdown(f"""
                **Texto:** {error['text'][:150]}...  
                **Real:** {error['true']} → **Predicho:** {error['pred']}  
                **Posible razón:** {error['reason']}
                """)

    # 7. Comparativa entre modelos
    with st.expander("🆚 Comparativa entre Modelos"):
        models = ["DistilBERT", "BERT", "T5"]
        fig = px.bar(
            x=models,
            y=[metrics[m]['f1'] for m in models],
            color=models,
            title="Comparación de F1-Score",
            labels={"x": "Modelo", "y": "F1-Score"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Conclusiones:**
        - BERT obtiene el mejor rendimiento general
        - DistilBERT ofrece mejor equilibrio entre rendimiento y eficiencia
        - T5 es más rápido pero menos preciso en esta tarea
        """)

    # 8. Justificación técnica
    with st.expander("🧠 Justificación Técnica"):
        st.markdown("""
        ### ¿Por qué estos resultados?
        
        **DistilBERT:**
        - Versión compacta de BERT con 40% menos parámetros
        - Mantiene el 97% del rendimiento de BERT
        - Ideal para despliegues con recursos limitados
        
        **BERT:**
        - Arquitectura bidireccional completa
        - Mayor capacidad de entender contexto
        - Requiere más recursos computacionales
        
        **T5:**
        - Modelo de tipo seq2seq
        - Bueno para generación de texto
        - Menos óptimo para clasificación binaria
        
        ### Limitaciones identificadas
        - Dificultad con lenguaje sarcástico/irónico
        - Errores en noticias con mezcla de hechos reales y falsos
        - Sensibilidad a dominios no vistos en entrenamiento
        """)
