from __future__ import annotations

from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout


def focal_binary_crossentropy(gamma: float = 2.0, alpha: float = 0.25):
    # Adapted focal loss for binary classification
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Compute basic BCE per example
        bce_loss = bce(y_true, y_pred)
        # Compute p_t
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce_loss)

    return loss


def build_mlp(
    input_dim: int,
    layers: List[int],
    dropout: Optional[List[float]] = None,
    batchnorm: bool = True,
    loss: str = "binary_crossentropy",
    optimizer: str = "adam",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
):
    model = Sequential()
    n_layers = len(layers)
    dropout = dropout or [0.0] * n_layers

    for idx, units in enumerate(layers):
        if idx == 0:
            model.add(Dense(units, activation="relu", input_dim=input_dim))
        else:
            model.add(Dense(units, activation="relu"))
        if batchnorm:
            model.add(BatchNormalization())
        dr = dropout[idx] if idx < len(dropout) else 0.0
        if dr and dr > 0:
            model.add(Dropout(dr))

    # output
    model.add(Dense(1, activation="sigmoid"))

    if loss == "focal":
        compiled_loss = focal_binary_crossentropy(gamma=focal_gamma, alpha=focal_alpha)
    else:
        compiled_loss = "binary_crossentropy"

    model.compile(optimizer=optimizer, loss=compiled_loss, metrics=["accuracy"])
    return model

