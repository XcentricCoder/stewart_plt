import numpy as np
import time
import matplotlib.pyplot as plt

# Example Ball Movement Simulation (Replace with real data)
class BallSimulation:
    def __init__(self):
        self.position_x = 0
        self.position_y = 0
        self.velocity_x = 0
        self.velocity_y = 0

    def update(self, force_x, force_y):
        self.velocity_x += force_x
        self.velocity_y += force_y
        self.position_x += self.velocity_x
        self.position_y += self.velocity_y

# Simulated Stewart Platform PID Tuning
def pid_tuning_test(Kp, Ki, Kd):
    pid = PIDController(Kp, Ki, Kd)
    sim = BallSimulation()
    errors_x, errors_y = [], []
    
    for t in range(100):  # Run for 100 iterations
        error_x = -sim.position_x  # Ball should be at (0,0)
        error_y = -sim.position_y
        control_signal = pid.compute(error_x, error_y, time.time())

        sim.update(control_signal[0], control_signal[1])
        
        errors_x.append(abs(error_x))
        errors_y.append(abs(error_y))

    avg_error = np.mean(errors_x) + np.mean(errors_y)
    return avg_error

# Grid search for best PID values
best_pid = None
lowest_error = float("inf")

for Kp in np.linspace(0.1, 3, 5):
    for Ki in np.linspace(0.01, 1, 5):
        for Kd in np.linspace(0.01, 1, 5):
            error = pid_tuning_test(Kp, Ki, Kd)
            print(f"Kp={Kp}, Ki={Ki}, Kd={Kd}, Error={error}")
            if error < lowest_error:
                lowest_error = error
                best_pid = (Kp, Ki, Kd)

print(f"Best PID values: Kp={best_pid[0]}, Ki={best_pid[1]}, Kd={best_pid[2]}")
