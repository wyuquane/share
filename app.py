from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import base64
from deepface import DeepFace

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set the maximum content length to 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Dummy user database
users = {
    "user1": {
        "username": "user1",
        "password": "password1",
        "face_encoding": None  # This should be updated with actual face encoding
    }
}

def decode_image(data_uri):
    header, encoded = data_uri.split(",", 1)
    data = base64.b64decode(encoded)
    np_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    image_data = request.form.get('image_data')

    if username in users and users[username]['password'] == password:
        # Perform face recognition
        captured_image = decode_image(image_data)
        try:
            result = DeepFace.verify(captured_image, users[username]['face_encoding'], enforce_detection=False)
            if result["verified"]:
                return redirect(url_for('home'))
            else:
                flash("Face recognition failed. Please try again.")
        except ValueError as e:
            flash(f"Error during face recognition: {str(e)}")
    else:
        flash("Invalid username or password.")
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        image_data = request.form.get('image_data')

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for('register'))

        if username in users:
            flash("Username already exists.")
            return redirect(url_for('register'))

        # Capture face image and encode
        captured_image = decode_image(image_data)
        try:
            face_encoding = DeepFace.represent(captured_image, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
            users[username] = {
                "username": username,
                "password": password,
                "face_encoding": face_encoding
            }
            flash("Registration successful. Please log in.")
            return redirect(url_for('index'))
        except ValueError as e:
            flash(f"Error during face encoding: {str(e)}")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
