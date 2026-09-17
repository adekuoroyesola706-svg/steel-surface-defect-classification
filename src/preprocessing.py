from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from .config import CLASS_NAMES, IMG_SIZE, BATCH_SIZE, SEED

def make_datagens(preprocess_fn):
    train = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )
    val_test = ImageDataGenerator(preprocessing_function=preprocess_fn)
    return train, val_test

def make_generator(datagen, df, shuffle):
    return datagen.flow_from_dataframe(
        df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        classes=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
