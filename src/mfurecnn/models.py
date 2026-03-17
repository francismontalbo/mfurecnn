"""Model builders for MFuRe-like multi-backbone classifiers."""

from __future__ import annotations

from tensorflow import keras

from .config import ModelConfig
from .layers import FusionResidualBlock


def _build_backbone(name: str, inputs, trainable: bool):
    builders = {
        "mobilenetv2": keras.applications.MobileNetV2,
        "resnet50": keras.applications.ResNet50,
        "efficientnetb0": keras.applications.EfficientNetB0,
    }
    if name not in builders:
        raise ValueError(f"Unsupported backbone: {name}")

    model = builders[name](include_top=False, weights="imagenet", input_tensor=inputs)
    model.trainable = trainable
    return model


def build_mfure_classifier(config: ModelConfig) -> keras.Model:
    """Build a production-friendly MFuRe-style classifier graph.

    This is a reusable code-path intended for script/CLI deployment while
    notebooks remain unchanged for paper-level experimentation.
    """

    inputs = keras.Input(shape=(*config.image_size, config.channels), name="image")

    mobile = _build_backbone("mobilenetv2", inputs, config.backbone_trainable).output
    resnet = _build_backbone("resnet50", inputs, config.backbone_trainable).output

    fused = FusionResidualBlock(filters=256, dropout_rate=config.dropout_rate)([mobile, resnet])
    pooled = keras.layers.GlobalAveragePooling2D()(fused)

    if config.use_alpha_dropout:
        pooled = keras.layers.AlphaDropout(config.dropout_rate)(pooled)
    else:
        pooled = keras.layers.Dropout(config.dropout_rate)(pooled)

    dense = keras.layers.Dense(config.dense_units, activation="selu")(pooled)
    outputs = keras.layers.Dense(config.num_classes, activation="softmax", name="predictions")(dense)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mfure_classifier")
    model.compile(optimizer=keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
