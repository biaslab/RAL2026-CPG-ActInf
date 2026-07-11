import numpy as np
import time
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.01
len_trial = 0.5
num_steps = int(len_trial / dt)
n_legs = 4

# Hopf oscillator parameters
alpha = 3
beta = 12
u = 2 
b = 150
coupling_gain = 8

k = np.array([
    [0, -1, -1, 1],
    [-1, 0, 1, -1],
    [-1, 1, 0, -1],
    [1, -1, -1, 0]
])
w_swing = 14; 
w_stance = 28; 

# Initialize states
x = np.zeros((n_legs, num_steps))
y = np.zeros((n_legs, num_steps))
theta = [0, np.pi/2, np.pi, 3*np.pi/2]
for i in range(4):
    x[i, 0] = np.sqrt(u) * np.cos(theta[i])
    y[i, 0] = np.sqrt(u) * np.sin(theta[i])

for step in range(1, num_steps):
    for j in range(n_legs):
        w = (w_stance / (np.exp(-b * y[j, step-1]) + 1)
             + w_swing / (np.exp(b * y[j, step-1]) + 1))
        r = np.sqrt(x[j, step-1]**2 + y[j, step-1]**2)
        
        x[j, step] = (x[j, step-1] + dt * (alpha * (u - r**2) * x[j, step-1]
                        - w * y[j, step-1]))
        
        coupling_term = coupling_gain * np.dot(k[j, :], y[:, step-1])
        y[j, step] = (y[j, step-1] + dt * (beta * (u - r**2) * y[j, step-1]
                        + w * x[j, step-1] + coupling_term))
        

fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 10))
# Use a color gradient for each leg based on time/progress
colors = [plt.cm.viridis(np.linspace(0, 1, x.shape[1])) for _ in range(4)]
sc0 = ax[0,0].scatter(x[0,:], y[0,:], c=np.linspace(0, 1, x.shape[1]), cmap='viridis', label='Leg 1')
sc1 = ax[1,0].scatter(x[1,:], y[1,:], c=np.linspace(0, 1, x.shape[1]), cmap='viridis', label='Leg 2')
sc2 = ax[0,1].scatter(x[2,:], y[2,:], c=np.linspace(0, 1, x.shape[1]), cmap='viridis', label='Leg 3')
sc3 = ax[1,1].scatter(x[3,:], y[3,:], c=np.linspace(0, 1, x.shape[1]), cmap='viridis', label='Leg 4')
# Add colorbars for each subplot
plt.colorbar(sc0, ax=ax[0,0], label='Time (Leg 1)')
plt.colorbar(sc1, ax=ax[1,0], label='Time (Leg 2)')
plt.colorbar(sc2, ax=ax[0,1], label='Time (Leg 3)')
plt.colorbar(sc3, ax=ax[1,1], label='Time (Leg 4)')

fig.savefig("figures/hopf_oscillator.png")