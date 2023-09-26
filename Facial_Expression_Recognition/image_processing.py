import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("model.h5")

def predict_expression(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (48, 48))
    normalized_image = resized_image / 255.0

    input_data = np.reshape(normalized_image, (1, 48, 48))

    emotion_labels = ["Disgust", "Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    prediction = model.predict(input_data)
    predicted_expression = emotion_labels[np.argmax(prediction)]

    print(predicted_expression)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    cv2.imshow("image", frame)

    key = cv2.waitKey(0q) & 0xFF

    if key == ord('x'):
        break

    if key == ord('q'):
        cv2.destroyAllWindows()
        break

    predict_expression(frame)

cv2.destroyAllWindows()