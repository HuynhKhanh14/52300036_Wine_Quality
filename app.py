import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from baseline_models import get_baseline_models
from neural_models import CNN_CONFIGS, LSTM_CONFIGS, build_cnn, build_lstm, set_seed
from preprocessing import FEATURES, CLASS_NAMES, load_data, split_data, to_sequence

st.set_page_config(page_title="Wine Quality ML Final Project", page_icon="🍷", layout="wide")
ROOT = Path(__file__).resolve().parent

st.markdown("""
<style>
.main-title {font-size: 40px; font-weight: 700;}
.sub-title {font-size: 18px; color: #666;}
</style>
""", unsafe_allow_html=True)


def metrics_from_pred(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


@st.cache_resource(show_spinner=False)
def train_neural_cached(wine_type, model_type, config_name):
    df = load_data(wine_type)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_data(df)
    configs = LSTM_CONFIGS if model_type == "LSTM" else CNN_CONFIGS
    config = next(c for c in configs if c["name"] == config_name)
    set_seed(42)
    model = build_lstm(config) if model_type == "LSTM" else build_cnn(config)
    X_train = to_sequence(X_train)
    X_val = to_sequence(X_val)
    X_test = to_sequence(X_test)
    start = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=config["batch_size"],
        callbacks=[
            __import__("tensorflow").keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            __import__("tensorflow").keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ],
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    return model, history.history, pred, y_test, elapsed, scaler


def baseline_result(wine_type, model_name):
    df = load_data(wine_type)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_data(df)
    models = get_baseline_models()
    model = models[model_name]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred, y_test, scaler


st.markdown('<div class="main-title">🍷 Wine Quality — Machine Learning Final Project</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Baseline ML + LSTM + CNN 1D | Train / Validation / Test | Hyperparameter experiments</div>', unsafe_allow_html=True)
st.divider()

st.sidebar.title("⚙️ Cài đặt")
wine_type = st.sidebar.selectbox("Loại rượu", ["Red Wine", "White Wine"])
page = st.sidebar.radio("Chức năng", [
    "📊 Tổng quan & EDA",
    "🤖 Baseline Models",
    "🧠 LSTM / GRU",
    "🧩 CNN 1D",
    "🏆 So sánh thực nghiệm",
    "🔮 Dự đoán mẫu mới",
])

df = load_data(wine_type)

if page == "📊 Tổng quan & EDA":
    st.header("📊 Tổng quan dữ liệu và tiền xử lý")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số mẫu", len(df))
    c2.metric("Số đặc trưng", len(FEATURES))
    c3.metric("Quality trung bình", f"{df.quality.mean():.2f}")
    c4.metric("Missing values", int(df[FEATURES + ["quality"]].isna().sum().sum()))
    st.info("Chia dữ liệu được dùng cho mô hình: 70% train, 15% validation, 15% test. StandardScaler chỉ được fit trên train để tránh data leakage.")
    st.subheader("Phân bố Quality")
    counts = df["category"].value_counts().reindex(CLASS_NAMES).fillna(0).reset_index()
    counts.columns = ["Category", "Count"]
    st.plotly_chart(px.bar(counts, x="Category", y="Count", text="Count", title=f"{wine_type} — phân bố 3 lớp"), use_container_width=True)
    st.subheader("Ma trận tương quan")
    st.plotly_chart(px.imshow(df[FEATURES + ["quality"]].corr(), text_auto=True, aspect="auto"), use_container_width=True)
    st.subheader("Dữ liệu mẫu")
    st.dataframe(df, use_container_width=True, height=350)

elif page == "🤖 Baseline Models":
    st.header("🤖 Baseline Machine Learning")
    st.write("Các baseline được giữ lại để so sánh công bằng với LSTM và CNN.")
    model_name = st.selectbox("Chọn baseline", list(get_baseline_models().keys()))
    model, pred, y_test, scaler = baseline_result(wine_type, model_name)
    report_df = metrics_from_pred(y_test, pred)
    st.dataframe(report_df, use_container_width=True)
    cm = confusion_matrix(y_test, pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, x=CLASS_NAMES, y=CLASS_NAMES, labels=dict(x="Predicted", y="Actual"), title=f"Confusion Matrix — {model_name}"), use_container_width=True)
    st.caption("Precision, Recall và F1-score dùng weighted average trong bảng tổng hợp để phù hợp khi phân bố lớp không cân bằng.")

elif page in ["🧠 LSTM / GRU", "🧩 CNN 1D"]:
    model_type = "LSTM" if page == "🧠 LSTM / GRU" else "CNN"
    configs = LSTM_CONFIGS if model_type == "LSTM" else CNN_CONFIGS
    st.header(f"{'🧠 LSTM' if model_type == 'LSTM' else '🧩 CNN 1D'}")
    st.write("Dữ liệu bảng gồm 11 thuộc tính được reshape thành sequence dài 11 với 1 feature mỗi bước để thực nghiệm theo yêu cầu môn học.")
    config_name = st.selectbox("Cấu hình thực nghiệm", [c["name"] for c in configs])
    config = next(c for c in configs if c["name"] == config_name)
    st.json(config)
    with st.spinner("Đang huấn luyện mô hình... lần đầu có thể mất vài phút"):
        model, history, pred, y_test, elapsed, scaler = train_neural_cached(wine_type, model_type, config_name)
    report_df = metrics_from_pred(y_test, pred)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{report_df.loc['accuracy', 'precision'] * 100:.2f}%")
    c2.metric("F1 weighted", f"{report_df.loc['weighted avg', 'f1-score'] * 100:.2f}%")
    c3.metric("Training time", f"{elapsed:.2f}s")
    st.subheader("Training history")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["loss"], mode="lines", name="Train Loss"))
    fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(title="Loss theo epoch", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["accuracy"], mode="lines", name="Train Accuracy"))
    fig.add_trace(go.Scatter(y=history["val_accuracy"], mode="lines", name="Validation Accuracy"))
    fig.update_layout(title="Accuracy theo epoch", xaxis_title="Epoch", yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Classification Report")
    st.dataframe(report_df, use_container_width=True)
    cm = confusion_matrix(y_test, pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, x=CLASS_NAMES, y=CLASS_NAMES, labels=dict(x="Predicted", y="Actual"), title=f"Confusion Matrix — {config_name}"), use_container_width=True)
    final_train_acc = history["accuracy"][-1]
    final_val_acc = history["val_accuracy"][-1]
    if final_train_acc - final_val_acc > 0.08:
        st.warning(f"Có dấu hiệu overfitting: train accuracy {final_train_acc:.3f} cao hơn validation accuracy {final_val_acc:.3f}.")
    elif final_train_acc < 0.70 and final_val_acc < 0.70:
        st.info("Kết quả cho thấy mô hình có thể đang underfitting; có thể tăng độ phức tạp hoặc điều chỉnh learning rate.")
    else:
        st.success("Chưa thấy dấu hiệu overfitting mạnh theo chênh lệch accuracy cuối quá trình học.")

elif page == "🏆 So sánh thực nghiệm":
    st.header("🏆 So sánh các mô hình")
    st.write("Để có bảng thực nghiệm chính thức cho báo cáo, chạy `python train_models.py --wine red` hoặc `--wine white`.")
    result_path = ROOT / "outputs" / ("red" if wine_type == "Red Wine" else "white") / "all_results.csv"
    if result_path.exists():
        results = pd.read_csv(result_path)
        display_cols = [c for c in ["Model", "Type", "Learning rate", "Batch size", "Epochs run", "Accuracy", "Precision", "Recall", "F1-score", "Training time (s)"] if c in results.columns]
        st.dataframe(results[display_cols].sort_values("F1-score", ascending=False), use_container_width=True)
        st.subheader("F1-score comparison")
        st.plotly_chart(px.bar(results, x="Model", y="F1-score", color="Type", text_auto=".3f", title=f"So sánh F1 — {wine_type}"), use_container_width=True)
    else:
        st.warning("Chưa có file kết quả. Hãy chạy train_models.py để tạo kết quả thực nghiệm.")

elif page == "🔮 Dự đoán mẫu mới":
    st.header("🔮 Dự đoán chất lượng rượu")
    model_family = st.selectbox("Loại mô hình", ["Logistic Regression", "K-Nearest Neighbors", "Gaussian Naive Bayes", "Decision Tree", "LSTM", "CNN"])
    defaults = dict(zip(FEATURES, df[FEATURES].median()))
    values = {}
    cols = st.columns(3)
    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            values[feature] = st.number_input(feature, value=float(defaults[feature]))
    if st.button("🔮 Dự đoán", type="primary"):
        sample = pd.DataFrame([values], columns=FEATURES)
        if model_family in get_baseline_models():
            model, _, _, scaler = baseline_result(wine_type, model_family)
            prediction = int(model.predict(sample)[0])
            label = CLASS_NAMES[prediction]
        else:
            config = (LSTM_CONFIGS if model_family == "LSTM" else CNN_CONFIGS)[1]
            model, _, _, _, _, scaler = train_neural_cached(wine_type, model_family, config["name"])
            x = scaler.transform(sample).astype("float32")
            prediction = int(np.argmax(model.predict(to_sequence(x), verbose=0), axis=1)[0])
            label = CLASS_NAMES[prediction]
        st.success(f"Kết quả dự đoán: **{label}**")

st.divider()
st.caption("Wine Quality Final Project | Python + Streamlit + Scikit-learn + TensorFlow/Keras + Plotly")
