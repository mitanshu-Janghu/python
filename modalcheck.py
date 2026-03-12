import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model Loaded")
print("Input shape:", input_details[0]['shape'])

# Emotion labels (change according to your model)
labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Load detectors
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

cap = cv2.VideoCapture(0)

sleep_counter = 0
attention_score = 100

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    state = "No Face"

    if len(faces) > 0:

        (x,y,w,h) = faces[0]

        face = gray[y:y+h,x:x+w]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Emotion detection
        img = cv2.resize(face,(48,48))
        img = img / 255.0
        img = np.reshape(img,(1,48,48,1)).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'],img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        emotion_id = np.argmax(output)
        emotion = labels[emotion_id]

        # Eye detection
        eyes = eye_cascade.detectMultiScale(face,1.3,5)

        if len(eyes) == 0:
            sleep_counter += 1
        else:
            sleep_counter = 0

        if sleep_counter > 20:
            state = "Sleepy"
            attention_score -= 1
        else:
            state = "Focused"
            attention_score += 0.5

        # Yawn detection
        mouths = mouth_cascade.detectMultiScale(face,1.5,20)

        if len(mouths) > 0:
            state = "Yawning"
            attention_score -= 1

        # Looking away
        if x < 50 or x+w > frame.shape[1]-50:
            state = "Looking Away"
            attention_score -= 1

        if attention_score < 0:
            attention_score = 0

        # Display emotion
        cv2.putText(frame,f"Emotion: {emotion}",
                    (x,y-40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,0,0),2)

    # Display state
    cv2.putText(frame,f"State: {state}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    # Display attention
    cv2.putText(frame,f"Attention: {attention_score}%",
                (30,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,255),2)

    cv2.imshow("AI Attention Monitor",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()