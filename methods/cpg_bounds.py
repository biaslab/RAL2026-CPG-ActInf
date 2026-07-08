"""Shared 8D CPG parameter bounds used by every optimizer.

Order: [coupling_gain, w_swing, w_stance, F_FAST, STOP_GAIN,
        hip_amplitude, knee_amplitude, b].
"""

import torch

bounds = torch.tensor([
    [4.0, 10.0, 10.0, 25.0, 0.05, 0.10, 0.5, 0.1],   # lower
    [12.0, 25.0, 25.0, 60.0, 0.5, 0.35, 1.0, 10.0],  # upper
], dtype=torch.double)

bounds_lower = bounds[0]
bounds_upper = bounds[1]
