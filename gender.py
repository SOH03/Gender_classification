import cv2
import numpy as np
from mtcnn import MTCNN
from keras.applications import VGG19
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.applications.vgg19 import preprocess_input
from mtcnn import MTCNN

# -------------------------------
# 1️⃣ Rebuild model
# -------------------------------
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256,256,3))
vgg.trainable = False

x = Flatten()(vgg.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=vgg.input, outputs=output)

# -------------------------------
# 2️⃣ Load saved weights
# -------------------------------
model.load_weights(r"F:\MLcode\compuerVISION\gender.weights.h5",  skip_mismatch=True)

# -------------------------------
# 3️⃣ MTCNN detector
detector = MTCNN()

# -------------------------------
# 4️⃣ Prediction helper function
def predict_gender(face_image):

    crop = cv2.resize(face_image, (256,256))
    crop = preprocess_input(np.expand_dims(crop.astype(np.float32), axis=0))
    pred = model.predict(crop)[0][0]
    if pred > 0.5:
        return "Male", round(pred*100,2)
    else:
        return "Female", round((1-pred)*100,2)

# -------------------------------
# 5️⃣ Webcam loop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)
    # Detect faces

    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0,x), max(0,y)
        face_crop = frame[y:y+h, x:x+w]
        
        try:
            label, conf = predict_gender(face_crop)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        except:
            continue

    cv2.imshow("Gender Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
