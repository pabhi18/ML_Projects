from flask import Flask, request
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model("model.h5") 

app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict_expression():
    frame = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(gray_image, (48, 48))

    normalized_image = resized_image / 255.0

    input_data = np.reshape(normalized_image, (1, 48, 48, 1))

    emotion_labels = ["Disgust", "Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    prediction = model.predict(input_data)

    predicted_expression = emotion_labels[np.argmax(prediction)]

    return predicted_expression, 200

if __name__ == "__main__":
    app.run(debug=True)