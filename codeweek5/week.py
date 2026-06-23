import cv2  
import numpy as np
import os

image = cv2.imread('image.jpg')  # Load an image from file
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Apply Gaussian blur to the grayscale image
adaptive_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Apply adaptive thresholding
contours, hierarchy = cv2.findContours(adaptive_threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the thresholded image

print ("Number of contours found:", len(contours))  # Print the number of contours found
for i,c in enumerate(contours):  # Loop through each contour
    area = cv2.contourArea(c)  # Calculate the area of the contour
    perim = cv2.arcLength(c,True)  # Calculate the perimeter of the first contour
    if area > 100000:
            continue  # If the area is greater than 100000, skip to the next contour
    if area < 500:
            continue  # If the area is less than 500, skip to the next contour
    if area > 100:  # If the area is greater than 100
        print(f"Contour {i} is significant with area {area}")  # Print that the contour is significant
    M = cv2.moments(c)  # Calculate the moments of the contour
    if M["m00"] != 0:  # Check if the zeroth moment is  
        cX = int(M["m10"] / M["m00"])  # Calculate the x-coordinate of the centroid
        cY = int(M["m01"] / M["m00"])  # Calculate the y-coordinate of the centroid
        print(f"Centroid of contour {i}: ({cX}, {cY})")  # Print the coordinates of the centroid
    else:
        print(f"Contour {i} has zero area, cannot compute centroid.")  # Print a message if the contour has zero area

    x, y, w, h = cv2.boundingRect(c)  # Get the bounding rectangle of the contour
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the bounding rectangle on the original image

    roi = image[y:y+h, x:x+w]  # Extract the region of interest (ROI) from the original image
    cv2.imwrite(f'roi_{i}.jpg', roi)  # Save the ROI as a separate image file    

cv2.imshow('Contours', image)  # Display the original image with contours
cv2.imshow('Adaptive Threshold', adaptive_threshold)  # Display the adaptive thresholded image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows