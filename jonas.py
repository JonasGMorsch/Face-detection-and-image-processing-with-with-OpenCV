import cv2
import numpy as np
import time

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the hat image with alpha channel
hat_img = cv2.imread('hat.png', -1)

# def add_hat(image, hat_img, face):
#     # Calculate the angle of the head
#     (x, y, w, h) = face
#     center_x = x + w // 2
#     center_y = y + h // 2

#     # Calculate the position for the hat
#     hat_width = w
#     hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
#     hat_resized = cv2.resize(hat_img, (hat_width, hat_height))

#     # Calculate the position to place the hat
#     x = center_x - hat_width // 2
#     y = center_y - h - hat_height // 2

#     # Add the hat to the image
#     for i in range(hat_height):
#         for j in range(hat_width):
#             if hat_resized[i, j, 3] != 0:  # Check if the pixel is not transparent
#                 image[y + i, x + j] = hat_resized[i, j, :3]

#     return image


def add_hat(frame, hat_img, face_cascade):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=5, minSize=(30, 30))
    #faces = face_cascade.detectMultiScale( gray, scaleFactor=1.25, minNeighbors=10,minSize=(100, 100)) 
    faces = face_cascade.detectMultiScale( gray, scaleFactor=1.25, minNeighbors=6,minSize=(100, 100))

    for face in faces:
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

        # # Add the hat to the image
        # for i in range(hat_height):
        #     for j in range(hat_width):
        #         if hat_resized[i, j, 3] != 0:  # Check if the pixel is not transparent
        #             frame[y + i, x + j] = hat_resized[i, j, :3]
        
        # Ensure the hat fits within the frame boundaries
        y1, y2 = max(0, y), min(frame.shape[0], y + hat_height)
        x1, x2 = max(0, x), min(frame.shape[1], x + hat_width)
        hat_y1, hat_y2 = max(0, -y), min(hat_height, frame.shape[0] - y)
        hat_x1, hat_x2 = max(0, -x), min(hat_width, frame.shape[1] - x)
        
        # Add the hat to the image using vectorized operations
        alpha_hat = hat_resized[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255.0
        alpha_frame = 1.0 - alpha_hat
        
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_hat * hat_resized[hat_y1:hat_y2, 
                 hat_x1:hat_x2, c] + alpha_frame * frame[y1:y2, x1:x2, c])
    return frame


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
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 60)

# Create a resizable window
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)


# Define the target frame rate
target_fps = 60
frame_duration = 1.0 / target_fps

# Initialize the counter and start time
counter = 0
start_time = time.time()


last_window_width = 0
last_window_height = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the current window size
    window_width, window_height = cv2.getWindowImageRect('Frame')[2:]  
    
    if window_width != last_window_width or window_height != last_window_height:
        # Create background with the desired resolution
        background = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        print("WINDOW RESIZED!")
    
    last_window_width = window_width
    last_window_height = window_height



    frame = add_hat(frame, hat_img, face_cascade)


    frame_resized = resize_with_aspect_ratio(
        frame, width=window_width,height=window_height)
    height, width = frame_resized.shape[:2]
    
    # Calculate the position to place the resized image
    x_offset = (window_width - width) // 2
    y_offset = (window_height - height) // 2
    
    # Place the resized image onto the new background
    background[y_offset:y_offset+height, x_offset:x_offset+width] = frame_resized[:, :, :3]

    # Display the final image
    cv2.imshow('Frame', background)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # FPS COUNTER
    elapsed_time = time.time() - start_time
    counter += 1 
    # Check if one second has passed
    if elapsed_time >= 1.0:
        # Print the counter value
        print(f'Loops per second: {counter}')
        
        # Reset the counter and start time
        counter = 0
        start_time = time.time()

cap.release()
cv2.destroyAllWindows()





































