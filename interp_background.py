import numpy as np
from PIL import Image

im = np.zeros([512, 512, 3])

c1 = np.array([0, 150, 255])
c2 = np.array([255, 126, 121])
c3 = np.array([115, 255, 121])
c4 = np.array([255, 252, 121])

steps = 512

for i in range(steps):
    ca = c1 + (i / steps) * (c2 - c1)
    cb = c3 + (i / steps) * (c4 - c3)
    for ii in range(steps):
        c = ca + (ii / steps) * (cb - ca)
        im[i, ii, :] = c

I = Image.fromarray(im.astype(np.uint8))

I.save('interp_back.png')



