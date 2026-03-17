"""Custom neural layers used in the MFuRe family."""

from __future__ import annotations

from tensorflow import keras


class FusionResidualBlock(keras.layers.Layer):
    """Self-normalizing residual fusion block inspired by the paper design.

    This layer expects two feature tensors with the same channel shape and merges
    them through concatenation + bottleneck projection + residual skip.
    """

    def __init__(self, filters: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.concat = keras.layers.Concatenate(name="fusion_concat")
        self.proj = keras.layers.Conv2D(filters, kernel_size=1, padding="same", activation="selu")
        self.norm = keras.layers.BatchNormalization()
        self.dropout = keras.layers.AlphaDropout(dropout_rate)
        self.residual = keras.layers.Conv2D(filters, kernel_size=1, padding="same")
        self.out_act = keras.layers.Activation("selu")

    def call(self, inputs, training=False):
        left, right = inputs
        merged = self.concat([left, right])
        projected = self.proj(merged)
        projected = self.norm(projected, training=training)
        projected = self.dropout(projected, training=training)

        skip = self.residual(left)
        return self.out_act(projected + skip)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "dropout_rate": self.dropout_rate})
        return config
