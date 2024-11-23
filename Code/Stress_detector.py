import cv2
import dlib
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from deepface import DeepFace
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate distance between eyebrows
def eye_brow_distance(leye, reye):
    distq = dist.euclidean(leye, reye)
    return distq

# Function to predict emotion from face ROI
def emotion_finder(faces, frame, emotion_classifier):
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    x, y, w, h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h, x:x+w]
    
    # Convert the region of interest (ROI) to grayscale
    roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    
    # Normalize and preprocess
    roi = roi.astype("float") / 255.0
    roi = np.expand_dims(roi, axis=-1)  # Add channel dimension for grayscale
    roi = np.expand_dims(roi, axis=0)   # Add batch dimension

    # Predict emotion
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    return label

# Normalize stress values and determine stress levels
def normalize_values(points, disp):
    if len(points) < 2:
        return 0, "Low Stress"
    min_point = min(points)
    max_point = max(points)
    if max_point == min_point:
        return 0, "Low Stress"
    normalized_value = abs(disp - min_point) / abs(max_point - min_point)
    stress_value = np.exp(-normalized_value)
    if stress_value >= 0.75:  # Adjust threshold based on testing
        return stress_value, "High Stress"
    else:
        return stress_value, "Low Stress"

# Initialize variables
ar_thresh = 0.3
eye_ar_consec_frame = 5
counter = 0
total = 0
points = []

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\somur\\OneDrive\\Desktop\\ab\\Stress-Detection\\shape_predictor_68_face_landmarks.dat")

# Load emotion classifier model
emotion_classifier = load_model("C:\\Users\\somur\\OneDrive\\Desktop\\ab\\Stress-Detection\\_mini_XCEPTION.102-0.66.hdf5", compile=False)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    clahe_image = np.uint8(clahe_image)

    # Detect faces in the grayscale image
    detections = detector(clahe_image, 0)

    for detection in detections:
        # Get facial landmarks
        shape = predictor(gray, detection)
        shape = face_utils.shape_to_np(shape)

        # Get the left and right eyes
        (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        left_eye = shape[lBegin:lEnd]
        right_eye = shape[rBegin:rEnd]

        # Calculate Eye Aspect Ratio (EAR)
        left_eye_EAR = eye_aspect_ratio(left_eye)
        right_eye_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_eye_EAR + right_eye_EAR) / 2.0

        # Blink Detection Logic
        if avg_EAR < ar_thresh:
            counter += 1
        else:
            if counter >= eye_ar_consec_frame:
                total += 1
            counter = 0

        # Emotion Detection using DeepFace
        dominant_emotion = emotion_finder(detection, frame, emotion_classifier)

        # Stress Detection based on eyebrow distance
        (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]

        distq = eye_brow_distance(leyebrow[-1], reyebrow[0])
        points.append(distq)
        
        # Keep only the last 50 points to avoid memory issues
        if len(points) > 50:
            points.pop(0)

        stress_value, stress_label = normalize_values(points, distq)

        # Draw red square box around the face
        x, y, w, h = face_utils.rect_to_bb(detection)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw the results on the frame
        cv2.putText(frame, f"Blinks: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Stress Level: {stress_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Stress Value: {int(stress_value * 100)}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw facial landmarks
        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Draw eyebrows
        leyebrowhull = cv2.convexHull(leyebrow)
        reyebrowhull = cv2.convexHull(reyebrow)
        cv2.drawContours(frame, [leyebrowhull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [reyebrowhull], -1, (255, 0, 0), 1)

    # Display the frame
    cv2.imshow("Stress and Emotion Detection", frame)

    # Exit on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Facial analysis complete.")