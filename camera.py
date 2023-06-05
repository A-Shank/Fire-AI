# imports
import torch
import numpy as np
import cv2
from time import time
import os

# Class to detect fire


class FireDetection:
    # Initialize the class
    def __init__(self, model_name):
        # Set the capture index to 0 which means it uses the default camera
        self.capture_index = 0
        # Load the model
        self.model = self.load_model(model_name)
        # Set the classes
        self.classes = self.model.names
        # Set the device to cuda if available else cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Print the device
        print("Using Device: ", self.device)

# Function to capture video
    def video_cap(self):
        # Return the video capture object
        return cv2.VideoCapture(self.capture_index)

# Function to load the model
    def load_model(self, model_name):
        # If model name is not none
        if model_name:
            # Load the needed scripts from github and use the custom model as provided.
            # Sadly this is needed like that otherwise i would have to submit the entire yolov5 repo due to dependencies.
            model = torch.hub.load(
                'ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        # Return the model
        return model
# Function to score the frame

    def score_frame(self, frame):
        # assign the model to the device
        self.model.to(self.device)
        # Frame to tensor
        frame = [frame]
        # Results is equal to the model with the frame
        results = self.model(frame)
        # labels and cords are equal to the results
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # Return labels and cords
        return labels, cord

# Function to convert class to label
    def class_to_label(self, x):
        # Return the class at the index of x
        return self.classes[int(x)]

# Function to plot the boxes
    def plot_boxes(self, results, frame):
        # labels and cords are equal to the results
        labels, cord = results
        # n is equal to the length of labels
        n = len(labels)
        # x_shape and y_shape are equal to the width and height of the frame
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        # For i in range of n
        for i in range(n):
            # row is equal to the cords at index i
            row = cord[i]
            # If the row at index 4 is greater than or equal to 0.3
            if row[4] >= 0.3:
                # x1, y1, x2, y2 are equal to the row at index 0, 1, 2, 3 multiplied by the x_shape and y_shape
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] *
                                                                                   x_shape), int(row[3] * y_shape)
                # color of the box
                bgr = (0, 255, 0)
                # Draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # Draw the text
                cv2.putText(frame, self.class_to_label(
                    labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        # Return the frame
        return frame
    # Function to call the class

    def __call__(self):
        # Set the capture to the video capture
        cap = self.video_cap()
        # Assert that the capture is opened
        assert cap.isOpened()
        # While true
        while True:
            # Read the capture
            ret, frame = cap.read()
            # Assert ret
            assert ret
            # Resize the frame
            frame = cv2.resize(frame, (1000, 1000))
            # Start time and end time
            start_time = time()
            # Results is equal to the score frame
            results = self.score_frame(frame)
            # Frame is equal to the plot boxes
            frame = self.plot_boxes(results, frame)
            # End time
            end_time = time()
            # FPS is equal to 1 divided by the end time minus the start time
            fps = 1 / np.round(end_time - start_time, 2)
            # Draw the text
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            # Show the frame
            cv2.imshow('Fire Detection', frame)
            # If the key is equal to q it ends programm
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the capture
        cap.release()


# Create a new object and execute.
detector = FireDetection(model_name='pt-file/best.pt')
# Call the object
detector()
