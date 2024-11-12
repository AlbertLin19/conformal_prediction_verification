import pickle
import cupy as cp
import cupyx as cpx
from tqdm import tqdm

# empirically verify Lemma 5 in Lin & Bansal (2024)
# p: underlying probability of violation
# N: number of calibration points
# k: number of violations in calibration set
# e: safety violation parameter
# b: confidence parameter
# D: number of simulations

e_candidates = cp.linspace(start=0, stop=1, num=int(1e3)) # the possible es that can be computed

def simulate_e_computation(p, N, b, D):
    scores = 1.0*(cp.random.uniform(low=0, high=1, size=D*N).reshape(D, N) < p) # 0 - nonviolation, 1 - violation
    ks = cp.sum(scores, axis=-1, keepdims=True)
    b_candidates = cpx.scipy.special.bdtr(ks, N, e_candidates[cp.newaxis])
    es = e_candidates[cp.argmax(b_candidates <= b, axis=-1)]
    return es # len(es) = D

ps = cp.linspace(start=0, stop=1, num=int(1e1))
Ns = [int(1e1), int(1e2), int(1e3)]
bs = cp.linspace(start=0, stop=1, num=int(1e1))
D = int(1e5)

data = cp.full((len(ps), len(Ns), len(bs), D), fill_value=cp.NaN)
for i, p in tqdm(enumerate(ps)):
    for j, N in tqdm(enumerate(Ns)):
        for k, b in tqdm(enumerate(bs)):
            es = simulate_e_computation(p, N, b, D)
            data[i, j, k] = es

with open('data_dict.pickle', 'wb') as f:
    pickle.dump({
        'ps': ps,
        'Ns': Ns,
        'bs': bs,
        'D': D,
        'data': data,
    }, f)