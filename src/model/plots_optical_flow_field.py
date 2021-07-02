# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

OUTPUT_DIR = 'data/results/plots/optflow_opencv_field.png'

flow = np.load('data/processed/flow/vtest/000001_000002.npy')

u = flow[:, :, 0]
v = flow[:, :, 1]

plt.imshow(np.sqrt(u**2 + v**2), cmap='gray')
plt.axis('off')
plt.savefig(OUTPUT_DIR)
plt.show()
plt.clf()
