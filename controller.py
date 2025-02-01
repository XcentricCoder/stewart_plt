# controller.py
import numpy as np
from ball_detection import detect_ball
from stewart_plt import StewartPlatform

def calculate_platform_movement(cX, cY, frame_width, frame_height):
    # Calculate error and generate translation/rotation
    center_x, center_y = frame_width // 2, frame_height // 2
    error_x = cX - center_x
    error_y = cY - center_y

    translation_gain = 1.0
    rotational_gain = 0.01

    trans = np.array([translation_gain * error_x, translation_gain * error_y, 0])
    rot = np.array([rotational_gain * error_y, rotational_gain * error_x, 0])

    return trans, rot

def control_platform(platform, frame):
    ball_pos = detect_ball(frame)
    if ball_pos:
        cX, cY = ball_pos
        trans, rot = calculate_platform_movement(cX, cY, frame.shape[1], frame.shape[0])
        platform.calculate_angles(trans, rot)
        return trans, rot
    return None, None
