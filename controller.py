#controller
import numpy as np
import time
from ball_detection import detect_ball
from stewart_plt import StewartPlatform

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        
        self.integral_x = 0
        self.integral_y = 0
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.prev_time = None

    def compute(self, error_x, error_y, current_time):
        # Compute time delta
        if self.prev_time is None:
            dt = 1.0
        else:
            dt = current_time - self.prev_time
        
        # Proportional terms
        Px = self.Kp * error_x
        Py = self.Kp * error_y
        
        # Integral terms (with anti-windup)
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        Ix = self.Ki * self.integral_x
        Iy = self.Ki * self.integral_y
        
        # Derivative terms
        Dx = self.Kd * (error_x - self.prev_error_x) / dt if dt > 0 else 0
        Dy = self.Kd * (error_y - self.prev_error_y) / dt if dt > 0 else 0
        
        # Compute outputs
        output_x = Px + Ix + Dx
        output_y = Py + Iy + Dy
        
        # Apply output limits
        if self.output_limits[0] is not None:
            output_x = max(output_x, self.output_limits[0])
            output_y = max(output_y, self.output_limits[0])
        if self.output_limits[1] is not None:
            output_x = min(output_x, self.output_limits[1])
            output_y = min(output_y, self.output_limits[1])
        
        # Update state
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_time = current_time
        
        return np.array([output_x, output_y, 0])

def calculate_platform_movement(cX, cY, frame_width, frame_height):
    # Calculate error from frame center
    center_x, center_y = frame_width // 2, frame_height // 2
    error_x = cX - center_x
    error_y = cY - center_y

    return error_x, error_y

def control_platform(platform, frame, pid_controller):
    # Current time for PID calculations
    current_time = time.time()
    
    # Detect ball position
    ball_pos = detect_ball(frame)
    
    if ball_pos:
        cX, cY = ball_pos
        
        # Calculate errors
        error_x, error_y = calculate_platform_movement(
            cX, cY, frame.shape[1], frame.shape[0]
        )
        
        # Compute PID control signals
        trans = pid_controller.compute(error_x, error_y, current_time)
        
        # Additional rotation calculation (optional)
        rot = np.array([
            0.01 * error_y,  # Pitch
            0.01 * error_x,  # Roll
            0
        ])
        
        # Update platform angles
        platform.calculate_angles(trans, rot)
        
        return trans, rot
    
    return None, None

