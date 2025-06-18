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
from transformers import TrainingArguments, Trainer
#-------------------------------------
import json
from datasets import load_dataset
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from huggingface_hub import hf_hub_download
# ——————————————————————————————————————
# Añade estos imports al inicio del archivo si no los tienes
from transformers import TrainingArguments, Trainer
import optuna

def load_model(model_type):
    # Simplemente devolvemos el modelo ya cargado en memoria
    _, model = models[model_type]
    return model

# Por ahora, usamos datasets de ejemplo o los mismos df de tu EDA:
train_dataset = load_dataset(..., split="train[:80%]")  # el 80% para train
val_dataset   = load_dataset(..., split="train[80%:]")  # el 20% para val

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}
# Función objective corregida
def objective(trial, model_type, metric):
    # 1. Definir parámetros de entrenamiento a optimizar
    train_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("epochs", 1, 5),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }
    
    # 2. Cargar modelo y configurar sus parámetros específicos
    model = load_model(model_type)
    
    # Configurar parámetros específicos del modelo
    if model_type == "DistilBERT":
        model.config.hidden_dropout_prob = trial.suggest_float("hidden_dropout", 0.1, 0.5)
    elif model_type == "BERT":
        model.config.attention_probs_dropout_prob = trial.suggest_float("attention_dropout", 0.1, 0.5)
    elif model_type == "T5":  # Corregido de "TS" a "T5"
        model.config.dropout_rate = trial.suggest_float("dropout", 0.1, 0.4)
    
    # 3. Configuración del entrenamiento
    training_args = TrainingArguments(
        output_dir=f"./results_{model_type}",
        eval_strategy="epoch",  # Corregido de "eval_strategy"
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        **train_params
    )
    
    # 4. Crear el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Asegúrate de definir esto
        eval_dataset=val_dataset,    # Asegúrate de definir esto
        compute_metrics=compute_metrics  # Asegúrate de definir esto
    )
    
    # 5. Entrenamiento y evaluación
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results[f"eval_{metric}"]
    
# This must be the FIRST Streamlit call!
st.set_page_config(page_title="📰 Fake News Detection", layout="wide")
# ——————————————————————————————————————
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Using device: {device}")
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
    # use official bert-base vocab
    bert_tok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert_mod  = BertForSequenceClassification.from_pretrained(bert_repo, subfolder=bert_subf).to(device)
    models["BERT"] = (bert_tok, bert_mod)
#
    return models
#
models = load_models()
#
# 3) Inference functions
def predict_t5(text: str):
    # 1) Unpack tokenizer + model + ids
    tok, mod, fake_id, real_id = models["T5"]

    # 2) Tokenization and send to device
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
# ————————————— SIDEBAR: NAVIGATION —————————————
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["1️⃣ Inference", "2️⃣ Dataset EDA", "3️⃣ Hyperparam Tuning", "4️⃣ Model Analysis"])

# ————————————— PAGE 1: INFERENCE INTERFACE —————————————
if page == "1️⃣ Inference":
    st.title("📰 Fake News Detection – Inference")
    st.markdown(
        "Enter news text and select the model from the sidebar."
    )
    model_choice = st.sidebar.selectbox("Model", ["T5", "DistilBERT", "BERT"])
    text_input   = st.text_area("🖋️ News text:", height=200)

    if st.button("🔍 Predict"):
        if not text_input.strip():
            st.warning("Please enter some text before predicting.")
        else:
            with st.spinner(f"Analyzing with {model_choice}..."):
                if model_choice == "T5":
                    label, conf = predict_t5(text_input)
                else:
                    label, conf = predict_cls(text_input, model_choice)
            st.subheader(f"**Prediction:** {label.upper()}")
            st.write("**Confidence scores:**")
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
            # Method 1: Load directly with datasets
            try:
                # Specify exact split we want
                dataset = load_dataset(
                    "ErfanMoosaviMonazzah/fake-news-detection-dataset-English",
                    split="train",  # Specify split directly
                    verification_mode="no_checks"  # Avoid checks that might fail
                )
                df = dataset.to_pandas()
                
                # Basic column verification
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
            # Method 2: Embedded sample data
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

    # Load and process data
    try:
        df = load_data()
        
        # Robust label conversion
        df["label"] = df["label"].apply(
            lambda x: 0 if str(x).lower() in ["fake", "false", "0", "no"] else 1
        )
        
        # Visualizations
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
        
        # Optional Word Cloud
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
    ## Optimization Process Configuration
    Fine-tuning model parameters to maximize performance
    """)

    # 1. Model selector and configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox(
            "Model to optimize",
            ["DistilBERT", "BERT", "T5"],
            help="Select which model to optimize"
        )
    
    with col2:
        n_trials = st.slider(
            "Number of trials",
            min_value=5,
            max_value=50,
            value=15,
            help="Number of experiments to run"
        )
    
    with col3:
        metric = st.selectbox(
            "Target metric",
            ["f1", "accuracy", "precision", "recall"],
            index=0,
            help="Primary metric to optimize"
        )

    # 2. Parameter panel with explanations
    with st.expander("📋 Parameter Optimization Details", expanded=True):
        st.markdown("""
        | Parameter          | Range               | Importance               |
        |--------------------|---------------------|---------------------------|
        | Learning Rate      | 1e-6 to 1e-4         | Controls learning speed |
        | Batch Size         | 8, 16, 32           | Affects memory and stability |
        | Epochs             | 1 to 5               | Prevents overfitting  |
        | Weight Decay       | 0.0 to 0.3           | L2 regularization         |
        | Dropout            | Model-specific       | Prevents overfitting      |
        """)

    # 3. Main optimization function
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner(f"Optimizing {model_type} - This may take several minutes..."):
            try:
                # Optuna configuration
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(),
                    pruner=optuna.pruners.HyperbandPruner()
                )
                
                # Run optimization
                study.optimize(
                    lambda trial: objective(trial, model_type, metric),
                    n_trials=n_trials,
                    show_progress_bar=True
                )

                # Show results
                st.success("Optimization completed!")
                
                # 4. Results visualization
                st.subheader("📊 Optimization Results")
                
                # Progress plot
                fig1 = px.line(
                    study.trials_dataframe(),
                    x="number",
                    y="value",
                    title=f"{metric} metric evolution",
                    labels={"value": metric, "number": "Trial"}
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Best parameters
                st.subheader("🎯 Best Parameters Found")
                best_params = study.best_params
                st.json(best_params)
                
                # Importance plot
                fig2 = optuna.visualization.plot_param_importances(study)
                st.pyplot(fig2)
                
                # 5. Results interpretation
                with st.expander("🔍 Results Interpretation"):
                    st.markdown(f"""
                    - **Best {metric} achieved**: {study.best_value:.4f}
                    - **Trial number**: {study.best_trial.number}
                    - **Key parameters**:
                        - Learning Rate: {best_params.get('learning_rate', 'N/A')}
                        - Batch Size: {best_params.get('per_device_train_batch_size', 'N/A')}
                    """)
                    
                    if model_type == "DistilBERT":
                        st.write("For DistilBERT, optimal dropout was:", best_params.get("hidden_dropout_prob", "N/A"))
                    elif model_type == "BERT":
                        st.write("For BERT, optimal attention dropout was:", best_params.get("attention_probs_dropout_prob", "N/A"))
                
                # 6. Option to save results
                if st.button("💾 Save Optimal Configuration"):
                    save_best_config(model_type, best_params)
                    st.toast("Configuration saved successfully!", icon="✅")
            
            except Exception as e:
                st.error(f"Optimization error: {str(e)}")
                st.error("Check logs for details")

# Optimization function (must be in same file or imported)
    
elif page == "4️⃣ Model Analysis":
    st.title("📊 Model Analysis")
    st.markdown("""
    ## Comparative Model Evaluation
    Detailed performance analysis of each architecture
    """)

    # 1. Initial setup
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        from transformers import pipeline
    except ImportError as e:
        st.error(f"Import error: {str(e)}")
        st.stop()

    # 2. Model selection for analysis
    model_type = st.radio(
        "Model to analyze",
        ["DistilBERT", "BERT", "T5"],
        horizontal=True
    )

    # 3. General metrics section
    with st.container():
        st.subheader("📈 General Performance")
        col1, col2, col3 = st.columns(3)
        
        # Example data (replace with your actual metrics)
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
            st.write("**Split:** Test (30% of total)")
            st.write(f"**Samples:** 12,543 articles")

    # 4. Confusion matrix
    with st.expander("🧮 Confusion Matrix", expanded=True):
        # Example data
        cm = np.array([[1250, 150], [120, 1280]]) if model_type == "DistilBERT" else \
             np.array([[1300, 100], [90, 1310]]) if model_type == "BERT" else \
             np.array([[1200, 200], [180, 1220]])
        
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fake', 'Real'], 
                    yticklabels=['Fake', 'Real'])
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {model_type}')
        st.pyplot(fig)

    # 5. Classification report
    with st.expander("📝 Detailed Classification Report"):
        # Example data
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

    # 6. Error analysis
    with st.expander("🔍 Error Analysis"):
        st.subheader("False Positive/Negative Examples")
        
        # Example data
        error_samples = {
            "DistilBERT": [
                {"text": "President announces new tax law...", "true": "Real", "pred": "Fake", "reason": "Political vocabulary"},
                {"text": "Miracle cancer cure discovered...", "true": "Fake", "pred": "Real", "reason": "Misused scientific language"}
            ],
            "BERT": [
                {"text": "8.5 magnitude earthquake hits coast...", "true": "Real", "pred": "Fake", "reason": "Extreme event"},
                {"text": "Celebrity reveals they're reptilian...", "true": "Fake", "pred": "Real", "reason": "Sensationalism"}
            ],
            "T5": [
                {"text": "New climate change study published...", "true": "Real", "pred": "Fake", "reason": "Technical terms"},
                {"text": "Vaccine causes autism, doctor claims...", "true": "Fake", "pred": "Real", "reason": "Pseudoscience"}
            ]
        }
        
        for error in error_samples[model_type]:
            with st.container(border=True):
                st.markdown(f"""
                **Text:** {error['text'][:150]}...  
                **True:** {error['true']} → **Predicted:** {error['pred']}  
                **Possible reason:** {error['reason']}
                """)

    # 7. Model comparison
    with st.expander("🆚 Model Comparison"):
        models = ["DistilBERT", "BERT", "T5"]
        fig = px.bar(
            x=models,
            y=[metrics[m]['f1'] for m in models],
            color=models,
            title="F1-Score Comparison",
            labels={"x": "Model", "y": "F1-Score"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Conclusions:**
        - BERT achieves best overall performance
        - DistilBERT offers better performance/efficiency balance
        - T5 is faster but less accurate for this task
        """)

    # 8. Technical justification
    with st.expander("🧠 Technical Justification"):
        st.markdown("""
        ### Why these results?
        
        **DistilBERT:**
        - Compact version of BERT with 40% fewer parameters
        - Maintains 97% of BERT's performance
        - Ideal for resource-constrained deployments
        
        **BERT:**
        - Full bidirectional architecture
        - Better context understanding
        - Requires more computational resources
        
        **T5:**
        - Seq2seq model type
        - Good for text generation
        - Less optimal for binary classification
        
        ### Identified limitations
        - Difficulty with sarcastic/ironic language
        - Errors with news mixing real and fake facts
        - Sensitivity to domains not seen in training
        """)
        
elif page == "5️⃣ Best Hyperparameters":
    st.title("🛠️ Best Hyperparameters Found")
    st.markdown("""
    These are the optimal hyperparameters that Optuna discovered during tuning.
    """)
    
    # 1) Cargamos el JSON
    try:
        with open("best_params.json", "r") as f:
            best_params = json.load(f)
    except FileNotFoundError:
        st.error("Cannot find `best_params.json` in the app folder.")
        st.stop()
    
    # 2) Mostramos la tabla de parámetros
    st.subheader("🔎 Parameters Overview")
    st.json(best_params)
    
    # 3) Métricas clave (si tu JSON incluye métricas adicionales, ajústalo aquí)
    #    En este caso asumimos sólo hiperparámetros, así que los mostramos como métricas
    st.subheader("📊 Params as Metrics")
    cols = st.columns(len(best_params))
    for (param, value), col in zip(best_params.items(), cols):
        col.metric(label=param, value=value)
