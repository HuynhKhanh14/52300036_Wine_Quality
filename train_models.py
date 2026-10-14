"""Reproducible experiment runner for all baseline, LSTM and CNN configurations.
Run: python train_models.py --wine red
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)

from baseline_models import run_all_baselines
from neural_models import CNN_CONFIGS, LSTM_CONFIGS, build_cnn, build_lstm, callbacks, set_seed
from preprocessing import load_data, split_data, to_sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def evaluate_neural(model, X_test, y_test):
    probs = model.predict(X_test, verbose=0)
    pred = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Confusion Matrix": confusion_matrix(y_test, pred).tolist(),
        "predictions": pred.tolist(),
    }


def train_one(model_type, config, X_train, y_train, X_val, y_val, X_test, y_test, out_dir):
    set_seed(42)
    builder = build_lstm if model_type == "LSTM" else build_cnn
    model = builder(config)
    start = perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=config["batch_size"],
        callbacks=callbacks(),
        verbose=0,
    )
    train_time = perf_counter() - start
    metrics = evaluate_neural(model, X_test, y_test)

    model_dir = out_dir / "models"
    figure_dir = out_dir / "figures"
    result_dir = out_dir / "results"
    model_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    model.save(model_dir / f"{config['name'].lower()}.keras")
    pd.DataFrame(history.history).to_csv(result_dir / f"{config['name'].lower()}_history.csv", index=False)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{config['name']} - Loss")
    plt.legend()
    plt.tight_layout()
    fig.savefig(figure_dir / f"{config['name'].lower()}_loss.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{config['name']} - Accuracy")
    plt.legend()
    plt.tight_layout()
    fig.savefig(figure_dir / f"{config['name'].lower()}_accuracy.png", dpi=160)
    plt.close(fig)

    row = {
        "Model": config["name"],
        "Type": model_type,
        "Learning rate": config["learning_rate"],
        "Batch size": config["batch_size"],
        "Epochs run": len(history.history["loss"]),
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-score": metrics["F1-score"],
        "Training time (s)": train_time,
        "Train final accuracy": history.history["accuracy"][-1],
        "Validation final accuracy": history.history["val_accuracy"][-1],
        "Train final loss": history.history["loss"][-1],
        "Validation final loss": history.history["val_loss"][-1],
        "Confusion Matrix": json.dumps(metrics["Confusion Matrix"]),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wine", choices=["red", "white"], default="red")
    args = parser.parse_args()

    wine_type = "Red Wine" if args.wine == "red" else "White Wine"
    out_dir = Path("outputs") / args.wine
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(wine_type)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_data(df)

    np.save(out_dir / "scaler_mean.npy", scaler.mean_)
    np.save(out_dir / "scaler_scale.npy", scaler.scale_)

    baseline_df = run_all_baselines(X_train, y_train, X_test, y_test)
    baseline_df["Type"] = "Baseline"
    baseline_df.to_csv(out_dir / "baseline_results.csv", index=False)

    X_train_seq = to_sequence(X_train)
    X_val_seq = to_sequence(X_val)
    X_test_seq = to_sequence(X_test)

    neural_rows = []
    for config in LSTM_CONFIGS:
        neural_rows.append(train_one("LSTM", config, X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test, out_dir))
    for config in CNN_CONFIGS:
        neural_rows.append(train_one("CNN", config, X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test, out_dir))

    neural_df = pd.DataFrame(neural_rows)
    neural_df.to_csv(out_dir / "neural_results.csv", index=False)

    combined = pd.concat([baseline_df, neural_df], ignore_index=True, sort=False)
    combined.to_csv(out_dir / "all_results.csv", index=False)
    print(combined[["Model", "Type", "Accuracy", "Precision", "Recall", "F1-score"]].to_string(index=False))
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
