import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from .config import CLASS_NAMES, IMG_SIZE

def predict(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    image = load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    array = img_to_array(image).astype("float32") / 255.0
    probs = model.predict(np.expand_dims(array, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    label, confidence = predict(args.model, args.image)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.3f}")
