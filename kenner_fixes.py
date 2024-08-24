import cv2
import tkinter
import threading
import numpy as np

# Global variables to keep track of states
binary_on = False
edge_detection_on = False
stop_video = True
laplacean_on = False
face_detect = True
hat_on = True
laplacean_k = np.array([[0 ,1 ,0],
                       [1, -4, 1],
                       [0, 1, 0]])

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the hat image with alpha channel
hat_img = cv2.imread('hat.png', -1)

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


def toggle_binarization():
    global binary_on
    binary_on = not binary_on

def toggle_hat():
    global hat_on
    hat_on = not hat_on

# Function to toggle laplacean edge detection
def toggle_laplacean():
    global laplacean_on
    laplacean_on = not laplacean_on

# Function to toggle edge detection
def toggle_edge_detection():
    global edge_detection_on
    edge_detection_on = not edge_detection_on

# Function to start or stop the video stream
def toggle_video_stream():
    global stop_video
    if stop_video:
        stop_video = False
        start_button.config(text="Stop Video")
        video_thread = threading.Thread(target=capture_and_display_video)
        video_thread.start()
    else:
        stop_video = True
        start_button.config(text="Start Video")

# Function to capture and display video
def capture_and_display_video():
    global stop_video

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize faces as an empty list
    #faces = []

    # Loop to continuously capture frames from the webcam
    while not stop_video:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        #Apply face detection
        #if face_detect:
            #faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                
        # Apply adaptive Gaussian thresholding if toggled on
        if binary_on:
            frame = cv2.adaptiveThreshold(
                gray_frame, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                15,  # Block size (must be odd)
                2    # Constant subtracted from the mean
            )
        #else:
            #frame = gray_frame

        # Apply Canny edge detection if toggled on
        if edge_detection_on:
            frame = cv2.Canny(frame, 100, 200)
            
         # Apply Laplacean edge detection if toggled on
        if laplacean_on:
            frame = cv2.filter2D(frame, -1, laplacean_k)


         # Apply hat if toggled on
        if hat_on:
            frame = add_hat(frame, hat_img, face_cascade)
            
            # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around the face
            #     frame = add_hat(frame, hat_img, (x, y, w, h))    
            
            
            #for face in faces: codigo original, descomentar depois
                #frame = add_hat(frame, hat_img, face)

                
        # Display the captured frame in a window
        cv2.imshow('Webcam Video', frame)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Function to handle closing the application
def on_closing():
    global stop_video
    stop_video = True  # Stop the video capture
    root.destroy()  # Close the Tkinter window

if __name__ == "__main__":
    # Create the main window
    root = tkinter.Tk()
    root.title("Webcam Video with Toggle")

    # Create a button to toggle binarization
    toggle_button = tkinter.Button(root, text="Toggle Binarization", command=toggle_binarization)
    toggle_button.pack()

    # Create a button to toggle edge detection
    edge_button = tkinter.Button(root, text="Toggle Edge Detection", command=toggle_edge_detection)
    edge_button.pack()

    # Create a button to toggle laplacean detection
    laplacean_button = tkinter.Button(root, text="Toggle Laplacean Detection", command=toggle_laplacean)
    laplacean_button.pack()

    # Create a button to start/stop the video stream
    hat_button = tkinter.Button(root, text="Habemus CHAPÉU", command=toggle_hat)
    hat_button.pack()

    # Create a button to start/stop the video stream
    start_button = tkinter.Button(root, text="Start Video", command=toggle_video_stream)
    start_button.pack()  

    
    # Bind the closing event to the on_closing function
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the Tkinter event loop
    root.mainloop()
