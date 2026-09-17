import numpy as np
import tensorflow as tf
import gradio as gr

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]
IMG_SIZE = (224, 224)
MODEL_PATH = "models/resnet50_final.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_defect(img):
    img = tf.image.resize(img, IMG_SIZE).numpy().astype("float32") / 255.0
    probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

demo = gr.Interface(
    fn=predict_defect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="Steel Surface Defect Classifier",
    description="Classify a steel-surface image into one of six NEU-DET defect categories.",
)

if __name__ == "__main__":
    demo.launch()
