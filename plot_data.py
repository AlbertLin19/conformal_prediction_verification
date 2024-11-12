import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
cmap = 'RdYlGn'
fontsize = 8

with open('data_dict.pickle', 'rb') as f:
    data_dict = pickle.load(f)
    
ps = data_dict['ps'].get()
Ns = data_dict['Ns']
bs = data_dict['bs'].get()
D = data_dict['D']
data = data_dict['data']

fig, axes = plt.subplots(2, len(Ns))
plt.suptitle(f'Observed - Desired Confidence ({D} Simulations)', size=fontsize)
for i, N in enumerate(Ns):
    ax = axes[0, i]
    ax.set_title(f'N={N}', size=fontsize)
    ax.set_xlabel('Underlying Prob of Violation', size=fontsize)
    ax.set_ylabel('Desired Confidence', size=fontsize)
    ax.set_box_aspect(1)
    
    Ps, Bs = np.meshgrid(ps, bs, indexing='ij')
    Os = np.mean(Ps[:, :, np.newaxis] <= data[:, i, :], axis=-1) # observed confidence
    Vs = Os-(1-Bs) # observed confidence - desired confidence

    legend_limit = np.max(np.abs(Vs))
    ax.pcolormesh(
        ps,
        1-bs,
        Vs.T,
        cmap=cmap,
        vmin=-legend_limit,
        vmax=+legend_limit,
    )
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=-legend_limit, vmax=+legend_limit), cmap=cmap), ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, i]
    ax.set_title('Claim is Valid (Black)', size=fontsize)
    ax.set_box_aspect(1)
    ax.pcolormesh(
        ps,
        1-bs,
        (Vs >= 0).T,
        cmap='binary',
        vmin=0,
        vmax=1,
    )

plt.tight_layout()
plt.savefig('data.png', dpi=800)