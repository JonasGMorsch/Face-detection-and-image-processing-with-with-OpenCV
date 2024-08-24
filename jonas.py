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


def resize_with_aspect_ratio(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
        
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
        
    else:
        r_w = width / float(w)
        r_h = height / float(h)
        r = min(r_w, r_h)
        dim = (int(w * r), int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# Open the video capture with DirectShow backend and set buffer size
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Set width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# Create a resizable window
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

while True:

    _, _, target_width, target_height = cv2.getWindowImageRect('Frame')  # Get the current window size
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # for face in faces:
    #     frame = add_hat(frame, hat_img, face)



    frame_resized = resize_with_aspect_ratio(
        frame, width=target_width,height=target_height)
        
    # Get the original resolution of the image
    height, width = frame_resized.shape[:2]
    
    # Calculate the position to place the resized image
    x_offset = (target_width - width) // 2
    y_offset = (target_height - height) // 2
    
    # Create a new image with the desired resolution
    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Place the resized image onto the new background
    background[y_offset:y_offset+height, x_offset:x_offset+width] = frame_resized[:, :, :3]

    # Display the final image
    cv2.imshow('Frame', background)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
exit()




































