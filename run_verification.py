import scipy
import numpy as np

# verify Pr^N({z^N: Gamma^e(z^N) >= 1-E}) >= 1-d, when e is computed as proposed by Lin & Bansal (2024)

# b: probability that z \in Q
# N: number of calibration points
def run_verification(b, N):
    scores = 1.0*(np.random.uniform(low=0, high=1, size=N) >= b)
    k = np.sum(N)

    d = np.linspace(0.0001, 0.9999, 1000)