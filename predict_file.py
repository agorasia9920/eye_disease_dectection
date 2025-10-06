# predict_file.py (fixed)
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "EyeModel.h5"
IMG_PATH = "Eye images/IMG-20250716-WA0126.jpg"  # Update with a real path
IMG_SIZE = (128, 128)  # Must match your model training size

class_labels = {0: 'Cataract', 1: 'Diabetic Retinopathy', 2: 'Glaucoma', 3: 'Normal'}

model = load_model(MODEL_PATH)

# Load and preprocess image
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Prediction
probs = model.predict(x)[0]
pred = np.argmax(probs)
confidence = np.max(probs)

print(f"Prediction: {class_labels[pred]} ({confidence*100:.2f}% confidence)")
