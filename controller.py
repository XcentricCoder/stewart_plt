import serial
import time
import numpy as np
from ball_detection import detect_ball
from stewart_plt import StewartPlatform

# Initialize Serial Connection (adjust COM port)
ser = serial.Serial('COM5', 115200, timeout=1)  # Change COM5 to your ESP32 port

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral_x = self.integral_y = 0
        self.prev_error_x = self.prev_error_y = 0
        self.prev_time = time.time()

    def compute(self, error_x, error_y):
        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time else 1.0

        # PID calculations
        Px, Py = self.Kp * error_x, self.Kp * error_y
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        Ix, Iy = self.Ki * self.integral_x, self.Ki * self.integral_y
        Dx = self.Kd * (error_x - self.prev_error_x) / dt if dt > 0 else 0
        Dy = self.Kd * (error_y - self.prev_error_y) / dt if dt > 0 else 0

        # Final control output
        output_x = Px + Ix + Dx
        output_y = Py + Iy + Dy

        # Update previous values
        self.prev_error_x, self.prev_error_y = error_x, error_y
        self.prev_time = current_time

        return np.array([output_x, output_y])

def control_platform(platform, frame, pid_controller):
    ball_pos = detect_ball(frame)
    if ball_pos:
        cX, cY = ball_pos
        frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
        error_x, error_y = cX - frame_center_x, cY - frame_center_y

        # Compute PID control signals
        trans = pid_controller.compute(error_x, error_y)
        
        # Convert translation to roll & pitch (simple scaling)
        roll, pitch = -0.01 * trans[0], 0.01 * trans[1]  

        # Send roll & pitch to ESP32 via Serial
        ser.write(f"{roll:.2f},{pitch:.2f}\n".encode())

        return roll, pitch
    return None, None

# Example initialization
pid = PIDController(Kp=1.5, Ki=0.05, Kd=0.7)

