# %%
from PIL import ImageChops
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi

flow = np.load('data/processed/flow/vtest/000001_000002.npy')

u = flow[:, :, 0]
v = flow[:, :, 1]

plt.imshow(np.sqrt(u**2 + v**2), cmap='gray',
           aspect='equal', extent=[0, 768, 0, 576])
plt.axis('off')
plt.margins(x=0)
plt.savefig('data/results/plots/optflow_opencv_field.png', bbox_inches='tight')
plt.show()
plt.clf()

# %%
img1 = Image.open('data/processed/_sequences/vtest/000001.png').convert('L')
img2 = Image.open('data/processed/_sequences/vtest/000008.png').convert('L')

plt.imshow(ImageChops.subtract(img2, img1), cmap='gray')
plt.axis('off')
plt.margins(x=0)
plt.savefig('data/results/plots/optflow_opencv_field_deduct.png',
            bbox_inches='tight')
plt.show()
plt.clf()
