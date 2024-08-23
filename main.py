import cv2
import tkinter as tk
from tkinter import Button
import threading

# Global variable to keep track of binarization state
binary_on = False

def toggle_binarization():
    global binary_on
    binary_on = not binary_on

def capture_and_display_video():
    # Open a connection to the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Loop to continuously capture frames from the webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive Gaussian thresholding if toggled on
        if binary_on:
            frame = cv2.adaptiveThreshold(
                gray_frame, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11,  # Block size (must be odd)
                2    # Constant subtracted from the mean
            )
        else:
            frame = gray_frame

        # Display the captured frame in a window
        cv2.imshow('Webcam Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def start_video_stream():
    # Run the OpenCV video capture in a separate thread
    video_thread = threading.Thread(target=capture_and_display_video)
    video_thread.start()

#if name == "main":
    # Create the main window
    root = tk.Tk()
    root.title("Webcam Video with Toggle")

    # Create a button to toggle binarization
    toggle_button = Button(root, text="Toggle Binarization", command=toggle_binarization)
    toggle_button.pack()

    # Create a button to start the video stream
    start_button = Button(root, text="Start Video", command=start_video_stream)
    start_button.pack()

    # Start the Tkinter event loop
    root.mainloop()
