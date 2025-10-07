import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from eye_check import is_eye_image
import tensorflow as tf
from datetime import datetime
import smtplib
from email.message import EmailMessage
import ssl

@st.cache_resource
def load_eye_model():
    return load_model('EyeModel.h5')
model = load_eye_model()

class_labels = {
    0: 'Cataract',
    1: 'Diabetic Retinopathy',
    2: 'Glaucoma',
    3: 'Normal'
}

recommendations = {
    ("Diabetic Retinopathy", None): {
        "advice": "Control blood sugar, regular eye check-ups every 6-12 months.",
        "tips": "Maintain healthy diet, manage blood pressure rigorously."
    },
    ("Glaucoma", None): {
        "advice": "Begin prescribed drops, regular follow-up with ophthalmologist.",
        "tips": "Take medication as prescribed, no missed doses."
    },
    ("Cataract", None): {
        "advice": "Monitor vision; see doctor if vision interferes with daily activities.",
        "tips": "Protect eyes from UV, wear sunglasses outdoors."
    },
    ("Normal", "None"): {
        "advice": "No disease detected. Maintain regular eye check-ups every 1-2 years.",
        "tips": "Eat a balanced diet, protect eyes from injury and UV light."
    }
}

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def send_email(to_email, subject, body, attachment=None):
    # Use Streamlit secrets (set in Cloud UI, not in code!)
    SENDER_EMAIL = st.secrets["email"]["user"]
    SENDER_PASSWORD = st.secrets["email"]["password"]
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg.set_content(body)
    if attachment is not None:
        maintype, subtype = attachment['mime_type'].split('/')
        msg.add_attachment(
            attachment['data'], maintype=maintype, subtype=subtype, filename=attachment['filename']
        )
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)

st.title("👁️ Eye Disease Detection (Retina Image)")
st.write("Upload or scan retinal image, review and edit advice before sending, attach files, and share an instant report!")

input_mode = st.radio("Choose image input method:", ["Upload from file", "Scan with camera"])
img = None

if input_mode == "Upload from file":
    uploaded_file = st.file_uploader("Upload image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
elif input_mode == "Scan with camera":
    img_file_buffer = st.camera_input("Take a retinal (fundus) picture")
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        st.image(img, caption="Captured Image", use_container_width=True)

if img is not None:
    if st.button("🔍 Predict"):
        if not is_eye_image(img):
            st.warning("⚠️ This image does not appear to be a valid retinal (fundus) image.")
        else:
            img_resized = image.img_to_array(img.resize((128, 128)))
            x = np.expand_dims(img_resized, axis=0)
            prediction = model.predict(x)
            pred_class = np.argmax(prediction)
            confidence = np.max(prediction)
            predicted_disease = class_labels[pred_class]
            predicted_severity = "None" if predicted_disease == "Normal" else None

            st.success(f"**{predicted_disease}** ({confidence*100:.2f}%)")

            key = (predicted_disease, predicted_severity)
            rec = recommendations.get(key, {"advice": "", "tips": ""})

            st.markdown("### 🩺 Personalized Clinical Recommendations")
            st.info(f"**Advice** (editable below): {rec['advice']}")
            st.write(f"**Lifestyle Tips** (editable below): {rec['tips']}")

            with st.expander("Show Model Explanation (Grad-CAM Heatmap)"):
                heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name='conv2d_3', pred_index=pred_class)
                heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size)
                heatmap_array = np.array(heatmap_resized)
                img_array = np.array(img.convert("RGB")).astype(np.float32)
                heatmap_color = cm.jet(heatmap_array / 255.0)[:, :, :3]
                superimposed_img = np.uint8(0.6 * img_array + 0.4 * 255 * heatmap_color)
                st.image(superimposed_img, caption="Grad-CAM Heatmap", use_container_width=True)

            with st.form("doctor_form"):
                advice = st.text_area("Advice to patient", value=rec['advice'])
                tips = st.text_area("Lifestyle tips", value=rec['tips'])
                prescription = st.text_area("Prescription (editable)")
                patient_email = st.text_input("Patient Email Address")
                uploaded_presc = st.file_uploader("Attach prescription/media (image/pdf)", type=["jpg", "jpeg", "png", "pdf"])
                submit_button = st.form_submit_button("Send Report to Patient")

            if submit_button and patient_email:
                email_body = f"""
Eye Disease Report ({datetime.now().strftime('%Y-%m-%d %I:%M %p')})

Diagnosis: {predicted_disease} ({confidence*100:.2f}%)
Severity: {predicted_severity}

Advice: {advice}
Lifestyle Tips: {tips}

Prescription:
{prescription}
"""
                attachment_info = None
                if uploaded_presc is not None:
                    attachment_info = {
                        "filename": uploaded_presc.name,
                        "data": uploaded_presc.getvalue(),
                        "mime_type": uploaded_presc.type or "application/octet-stream"
                    }
                try:
                    send_email(
                        to_email=patient_email,
                        subject="Your Eye Disease Report",
                        body=email_body,
                        attachment=attachment_info
                    )
                    st.success(f"✅ Report sent to {patient_email}")
                except Exception as e:
                    st.error(f"Email sending failed: {e}")

st.markdown("---")
st.caption("Demo only. Attachments must be small. Emails use Google SMTP; use app password and check spam/junk folders if not received.")
