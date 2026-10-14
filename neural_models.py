"""LSTM and 1D-CNN model definitions for the final ML project."""
from typing import Dict

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, BatchNormalization, Conv1D, Dense, Dropout,
    GlobalAveragePooling1D, MaxPooling1D,
)


def set_seed(seed: int = 42):
    tf.keras.utils.set_random_seed(seed)


def build_lstm(config: Dict) -> tf.keras.Model:
    model = Sequential(name=config["name"])
    model.add(tf.keras.Input(shape=(11, 1)))
    model.add(LSTM(config["units"], return_sequences=False))
    model.add(Dropout(config["dropout"]))
    if config.get("batch_norm", False):
        model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(config["dropout"]))
    model.add(Dense(3, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(config: Dict) -> tf.keras.Model:
    model = Sequential(name=config["name"])
    model.add(tf.keras.Input(shape=(11, 1)))
    model.add(Conv1D(config["filters1"], config["kernel_size"], activation="relu", padding="same"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(config["filters2"], config["kernel_size"], activation="relu", padding="same"))
    if config.get("batch_norm", False):
        model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(config["dropout"]))
    model.add(Dense(3, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0),
    ]


LSTM_CONFIGS = [
    {"name": "LSTM-1", "units": 32, "dropout": 0.20, "learning_rate": 1e-3, "batch_size": 32, "batch_norm": False},
    {"name": "LSTM-2", "units": 64, "dropout": 0.30, "learning_rate": 1e-3, "batch_size": 32, "batch_norm": False},
    {"name": "LSTM-3", "units": 128, "dropout": 0.50, "learning_rate": 5e-4, "batch_size": 64, "batch_norm": True},
]

CNN_CONFIGS = [
    {"name": "CNN-1", "filters1": 32, "filters2": 64, "kernel_size": 3, "dropout": 0.20, "learning_rate": 1e-3, "batch_size": 32, "batch_norm": False},
    {"name": "CNN-2", "filters1": 64, "filters2": 128, "kernel_size": 3, "dropout": 0.30, "learning_rate": 1e-3, "batch_size": 32, "batch_norm": True},
    {"name": "CNN-3", "filters1": 64, "filters2": 128, "kernel_size": 5, "dropout": 0.50, "learning_rate": 5e-4, "batch_size": 64, "batch_norm": True},
]
