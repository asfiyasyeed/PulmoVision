from flask import Flask, render_template, request, redirect, session
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os


app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load model (inside models folder)
model = load_model("models/pneumonia_model.h5")

# Login credentials
USERNAME = "hospital_admin"
PASSWORD = "pneumonia123"


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

    if file.filename == "":
        return redirect('/dashboard')

    # Save image
    image_path = os.path.join("static", file.filename)
    file.save(image_path)

    # ---- IMPORTANT: MATCH ORIGINAL TRAINING ---- #
    img = load_img(image_path,
                   target_size=(500, 500),
                   color_mode='grayscale')

    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediction
    classes = model.predict(x)
    result_value = classes[0][0]

    if result_value >= 0.5:
        prediction = "PNEUMONIA"
        confidence = round(result_value * 100, 2)
    else:
        prediction = "NORMAL"
        confidence = round((1 - result_value) * 100, 2)

    return render_template("result.html",
                           prediction=prediction,
                           confidence=confidence,
                           imagePath="/" + image_path)


if __name__ == '__main__':
    app.run(debug=True)