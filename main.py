import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_image_with_alpha(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 3:
        b, g, r = cv2.split(image)
        alpha = np.ones_like(b) * 255
        image = cv2.merge((b, g, r, alpha))
    return image

def alpha_blend(background, overlay):
    alpha = overlay[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    blended = (background * (1 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)
    return blended

eye_img = load_image_with_alpha('eye.png')
mouth_img = load_image_with_alpha('mouth.png')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        eye_x = x + w // 4
        eye_y = y + h // 4
        eye_w = w // 2
        eye_h = h // 4

        eye_resized = cv2.resize(eye_img, (eye_w, eye_h))

        frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w] = alpha_blend(frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w], eye_resized)

        mouth_x = x + w // 4
        mouth_y = y + h // 2
        mouth_w = w // 2
        mouth_h = h // 4

        mouth_resized = cv2.resize(mouth_img, (mouth_w, mouth_h))

        frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w] = alpha_blend(frame[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w], mouth_resized)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
