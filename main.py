import cv2
import numpy as np

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the hat image with alpha channel
hat_img = cv2.imread('hat.png', -1)

def add_hat(image, hat_img, face):
    # Calculate the angle of the head
    (x, y, w, h) = face
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the position for the hat
    hat_width = w
    hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
    hat_resized = cv2.resize(hat_img, (hat_width, hat_height))

    # Calculate the position to place the hat
    x = center_x - hat_width // 2
    y = center_y - h - hat_height // 2

    # Add the hat to the image
    for i in range(hat_height):
        for j in range(hat_width):
            if hat_resized[i, j, 3] != 0:  # Check if the pixel is not transparent
                image[y + i, x + j] = hat_resized[i, j, :3]

    return image

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for face in faces:
        frame = add_hat(frame, hat_img, face)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
