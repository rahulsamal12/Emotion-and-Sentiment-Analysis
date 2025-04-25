import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# Load models
face_classifier = cv2.CascadeClassifier('AIES-facial-Emotion-Recognition-ML-main/haarcascade_frontalface_default.xml')
classifier = load_model('AIES-facial-Emotion-Recognition-ML-main/model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_emojis = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😄', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
}

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

    # Face Mesh for Lining
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

    for i, (x, y, w, h) in enumerate(faces):
        if w < 50 or h < 50:
            continue

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        predictions = classifier.predict(roi)[0]
        max_index = np.argmax(predictions)
        label = emotion_labels[max_index]
        emoji = emotion_emojis[label]
        confidence = predictions[max_index]

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Text display
        text = f'{label} {emoji} ({int(confidence*100)}%)'
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw emoji with PIL
        try:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            font = ImageFont.truetype("seguiemj.ttf", 36)  # Adjust if needed
            draw.text((x + w + 10, y), emoji, font=font, fill=(255, 255, 255, 0))
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        except:
            pass

        # Emotion Probability Bar Graph
        for idx, (emotion, prob) in enumerate(zip(emotion_labels, predictions)):
            bar_width = int(prob * 100)
            cv2.rectangle(frame, (x, y + h + 25 + idx * 20), (x + bar_width, y + h + 40 + idx * 20), (100, 255, 100), -1)
            cv2.putText(frame, f'{emotion} {int(prob*100)}%', (x, y + h + 40 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow('🎭 Emotion Detector with Face Lining', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
