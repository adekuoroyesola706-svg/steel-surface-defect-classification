from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from .config import IMG_SIZE, CLASS_NAMES

def build_custom_cnn():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = inputs
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return models.Model(inputs, outputs, name="custom_cnn")

def build_transfer_model(base_model_fn, name):
    base = base_model_fn(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return models.Model(base.input, outputs, name=name), base

def build_mobilenetv2():
    return build_transfer_model(MobileNetV2, "mobilenetv2")

def build_resnet50():
    return build_transfer_model(ResNet50, "resnet50")
