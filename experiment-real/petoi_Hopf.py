from PetoiRobot import *
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import pdb

autoConnect()

dt = 0.01       # time step
T = 500         # number of steps
n_legs = 4

# Hopf oscillator parameters
alpha = 2.5
beta = 2.5
u = 2.0           # amplitude
b = 100.0         # sharpness of stance/swing transition

w_swing = 2
w_stance = 4


eff_freq = 6
#eff_freq = 5 for walk and 6 for trot 
w_stance *= eff_freq
w_swing  *= eff_freq

coupling_gain = 5.0

k_walk = np.array([[0, -1, 1, -1],
              [-1, 0, -1, 1],
              [-1, 1, 0, -1],
              [1, -1, -1, 0]])


k_trot = np.array([
    [0, -1, 1, -1],
    [-1, 0, -1, 1],
    [1, -1, 0, -1],
    [-1, 1, -1, 0]
])

k_matrix = k_trot

hip_offset = 40
hip_amplitude = 12
knee_offset = 30
knee_amplitude = 6

#best mapping so far hipoff = 40 hipampl = 12 knee_off = 30 kneeampl = 6 

# State arrays for oscillators
x = np.zeros((n_legs, T))
y = np.zeros((n_legs, T))
init_phases = [0, np.pi/2, np.pi, 3*np.pi/2]
for i in range(n_legs):
    x[i, 0] = np.sqrt(u) * np.cos(init_phases[i])
    y[i, 0] = np.sqrt(u) * np.sin(init_phases[i])

joint_hip_angles = np.zeros((T, n_legs))
joint_knee_angles = np.zeros((T, n_legs))

# Reset joints to neutral position
time.sleep(2.0)
joint_init = [8,0,12,0, 9,0,13,0, 10,0,14,0, 11,0,15,0]
send(goodPorts, ['I', joint_init, 0])
time.sleep(1.0)
#pdb.set_trace()
# Joint port mappings
hips  = [8, 9, 10, 11]
knees = [12, 13, 14, 15]

# Main loop
t0 = time.time()
for t in range(1, T):
    for j in range(n_legs):
        w = (w_stance / (np.exp(-b * y[j, t-1]) + 1)
             + w_swing / (np.exp(b * y[j, t-1]) + 1))
        r = np.sqrt(x[j, t-1]**2 + y[j, t-1]**2)
        x[j, t] = x[j, t-1] + dt * (alpha * (u - r**2) * x[j, t-1] - w * y[j, t-1])
        coup = coupling_gain * np.dot(k_matrix[j, :], y[:, t-1])
        y[j, t] = y[j, t-1] + dt * (beta * (u - r**2) * y[j, t-1] + w * x[j, t-1] + coup)

    for j in range(n_legs):
        hip_angle = hip_offset + hip_amplitude * x[j, t]
        knee_angle = knee_offset - knee_amplitude * max(0.0, y[j, t])
        joint_hip_angles[t, j] = hip_angle
        joint_knee_angles[t, j] = knee_angle

    joint_array = [8, int(joint_hip_angles[t,0]), 12, int(joint_knee_angles[t,0]),
                   9, int(joint_hip_angles[t,1]), 13, int(joint_knee_angles[t,1]),
                   10, int(joint_hip_angles[t,2]), 14, int(joint_knee_angles[t,2]),
                   11, int(joint_hip_angles[t,3]), 15, int(joint_knee_angles[t,3])]

    print(f'\r{t}/{T} {joint_array}', end='')
    send(goodPorts, ['I', joint_array, 0])
    time.sleep(0.005)

closePort()
print("\nSimulation complete in {:.2f}s".format(time.time() - t0))



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))
for j in range(n_legs):
    ax1.plot(range(T), joint_hip_angles[:, j], label=f'hip_{j+1}')
    ax2.plot(range(T), joint_knee_angles[:, j], label=f'knee_{j+1}')
ax1.set_title('Joint hip angles send'); ax1.legend()
ax2.set_title('Joint knee angles send'); ax2.legend()
plt.tight_layout()
plt.show()


fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8,6))
for j in range(n_legs):
    ax3.plot(range(T), x[j, :], label=f'x_{j+1}')
    ax4.plot(range(T), y[j, :], label=f'y_{j+1}')
ax3.set_title('CPG Output x over Time'); ax3.legend()
ax4.set_title('CPG Output y over Time'); ax4.legend()
plt.tight_layout()
plt.show()

