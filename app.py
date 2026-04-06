from flask import Flask, render_template, request, redirect, session, send_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
import numpy as np
import os
import io
import cv2
from datetime import datetime
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load model (inside models folder)
model = load_model("models/pneumonia_model.h5")

# Login credentials
USERNAME = "hospital_admin"
PASSWORD = "pneumonia123"


# ──────────────────────────────────────────
#  Grad-CAM helper
# ──────────────────────────────────────────
def generate_gradcam(img_array, model, last_conv_layer_name=None):
    """
    Generates a Grad-CAM heatmap overlaid on the original image.
    Automatically detects the last Conv2D layer if none is specified.
    Returns the path to the saved heatmap image.
    """
    # Auto-detect last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        return None  # No conv layer found — skip Grad-CAM silently

    # Build a model that outputs (last conv activations, final predictions)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
    model.get_layer(last_conv_layer_name).output,
    model.layers[-1].output
]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # For binary classification, use the single output neuron
        loss = predictions[:, 0]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return None  # 🔥 skip Grad-CAM safely

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to match original image (500x500 grayscale → RGB for overlay)
    heatmap_resized = cv2.resize(heatmap, (500, 500))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Load the original image as RGB for overlay
    orig_img = img_array[0]  # shape (500, 500, 1) grayscale
    orig_rgb = np.concatenate([orig_img, orig_img, orig_img], axis=-1)  # → (500,500,3)
    orig_rgb = np.uint8(orig_rgb * 255)

    # Superimpose
    superimposed = cv2.addWeighted(orig_rgb, 0.55, heatmap_color, 0.45, 0)

    # Save
    heatmap_filename = "gradcam_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    heatmap_path = os.path.join("static", heatmap_filename)
    cv2.imwrite(heatmap_path, superimposed)

    return heatmap_path


# ──────────────────────────────────────────
#  PDF report helper
# ──────────────────────────────────────────
def generate_pdf(prediction, confidence, image_path, heatmap_path, patient_id, scan_date):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # Header
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "PulmoVision", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Pneumonia Detection and Classification using Deep Learning", ln=True)

    pdf.line(20, pdf.get_y() + 4, 190, pdf.get_y() + 4)
    pdf.ln(10)

    # Report meta
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(50, 8, "Date of Analysis:", ln=False)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, scan_date, ln=True)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 8, "Patient ID:", ln=False)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, patient_id if patient_id else "N/A", ln=True)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 8, "Hospital:", ln=False)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "PulmoVision Hospital System", ln=True)

    pdf.ln(6)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # Diagnosis block
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Diagnosis Result", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 28)
    if prediction == "PNEUMONIA":
        pdf.set_text_color(180, 60, 40)
    else:
        pdf.set_text_color(40, 140, 100)
    pdf.cell(0, 14, prediction, ln=True)

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, f"Confidence Score: {confidence}%", ln=True)
    pdf.ln(6)

    # Confidence bar (drawn manually)
    bar_x, bar_y, bar_w, bar_h = 20, pdf.get_y(), 170, 6
    pdf.set_fill_color(230, 230, 230)
    pdf.rect(bar_x, bar_y, bar_w, bar_h, style="F")
    fill_w = bar_w * (confidence / 100)
    if prediction == "PNEUMONIA":
        pdf.set_fill_color(180, 60, 40)
    else:
        pdf.set_fill_color(40, 140, 100)
    pdf.rect(bar_x, bar_y, fill_w, bar_h, style="F")
    pdf.ln(14)

    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # Images
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Radiograph Analysis", ln=True)
    pdf.ln(4)

    img_y = pdf.get_y()
    img_w = 78

    # Original X-ray
    if os.path.exists(image_path.lstrip("/")):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.set_x(20)
        pdf.cell(img_w, 6, "Original X-Ray", ln=False)
        pdf.set_x(20 + img_w + 14)
        if heatmap_path and os.path.exists(heatmap_path):
            pdf.cell(img_w, 6, "Grad-CAM Heatmap", ln=True)
        else:
            pdf.ln()

        pdf.image(image_path.lstrip("/"), x=20, y=pdf.get_y(), w=img_w)

        if heatmap_path and os.path.exists(heatmap_path):
            pdf.image(heatmap_path, x=20 + img_w + 14, y=pdf.get_y(), w=img_w)

        pdf.ln(img_w + 4)

    pdf.ln(6)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # Model info
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Model Information", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, "Architecture: Convolutional Neural Network (CNN)", ln=True)
    pdf.cell(0, 7, "Classification Type: Binary (Pneumonia / Normal)", ln=True)
    pdf.cell(0, 7, "Input Size: 500 x 500 pixels (Grayscale)", ln=True)
    pdf.ln(8)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(
        0, 6,
        "DISCLAIMER: This report is generated by an AI-assisted tool and is intended for "
        "supplementary purposes only. It does not constitute a clinical diagnosis. Always "
        "consult a qualified radiologist or physician before making medical decisions."
    )

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(0, 6, f"PulmoVision  ·  Generated on {scan_date}  ·  Hospital Use Only", align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


# ──────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────
@app.route('/')
def home():
    return render_template("login.html")


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == USERNAME and password == PASSWORD:
        session['user'] = username
        return redirect('/dashboard')
    else:
        return render_template("login.html", error="Invalid Credentials")


@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template("dashboard.html")
    return redirect('/')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/')

    if 'imagefile' not in request.files:
        return redirect('/dashboard')

    file = request.files['imagefile']
    patient_id = request.form.get('patient_id', '').strip()

    if file.filename == "":
        return redirect('/dashboard')

    # Save image
    image_path = os.path.join("static", file.filename)
    file.save(image_path)

    # ---- MATCH ORIGINAL TRAINING ---- #
    img = load_img(image_path, target_size=(500, 500), color_mode='grayscale')
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediction
    classes = model.predict(x)
    result_value = float(classes[0][0])  # 🔥 FIX

    if result_value >= 0.5:
        prediction = "PNEUMONIA"
        confidence = round(result_value * 100, 2)
    else:
        prediction = "NORMAL"
        confidence = round((1 - result_value) * 100, 2)
    # Grad-CAM
    heatmap_path = generate_gradcam(x, model)
    heatmap_web_path = ("/" + heatmap_path) if heatmap_path else None

    # Store in session for PDF download
    session['last_result'] = {
        'prediction': prediction,
        'confidence': confidence,
        'image_path': "/" + image_path,
        'heatmap_path': heatmap_web_path,
        'patient_id': patient_id,
        'scan_date': datetime.now().strftime("%d %B %Y, %H:%M")
    }

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        imagePath="/" + image_path,
        heatmapPath=heatmap_web_path,
        patient_id=patient_id,
        scan_date=datetime.now().strftime("%d %B %Y, %H:%M")
    )


@app.route('/download-report')
def download_report():
    if 'user' not in session or 'last_result' not in session:
        return redirect('/')

    r = session['last_result']
    buf = generate_pdf(
        prediction=r['prediction'],
        confidence=r['confidence'],
        image_path=r['image_path'],
        heatmap_path=r['heatmap_path'].lstrip('/') if r['heatmap_path'] else None,
        patient_id=r['patient_id'],
        scan_date=r['scan_date']
    )

    filename = f"PulmoVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype='application/pdf',
                     as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True)