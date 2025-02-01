import cv2
import numpy as np

def detect_ball(frame):
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for the ball (e.g., yellow ball)
    lower_bound = np.array([20, 100, 100])  # Adjust as per your ball color
    upper_bound = np.array([40, 255, 255])  # Adjust as per your ball color

    # Create mask to isolate the ball
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply morphological operations to remove noise (optional)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close operation

    # Find contours of the ball
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours are found, calculate the center
    if contours:
        # Filter out small contours based on area
        contours = [c for c in contours if cv2.contourArea(c) > 500]

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the detected ball center on the frame (optional)
                cv2.circle(frame, (cX, cY), 10, (0, 255, 0), -1)

                return cX, cY
    return None

def detect_ball_with_hough(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Use HoughCircles to detect circles (balls)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle on the image
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            return x, y
    return None

# Initialize the video capture object (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Call either of the detection methods (detect_ball or detect_ball_with_hough)
    # Uncomment one of the following lines to use a specific method
    center = detect_ball(frame)  # Color-based detection
    # center = detect_ball_with_hough(frame)  # Hough Circle detection

    if center:
        print(f"Ball detected at: {center}")
    else:
        print("Ball not detected.")

    # Display the frame with the detection
    cv2.imshow("Ball Detection", frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
