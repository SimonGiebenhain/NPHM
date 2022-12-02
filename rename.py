import os
from PIL import Image
import numpy as np

root = '/Users/sigi/phd/test_project_page2/docs/static/latent_interpolation/car/'

files = [f for f in os.listdir(root) if f.endswith('.png')]

files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
print(files)


for i, f in enumerate(files):
    im1 = Image.open(root + f)
    a = np.array(im1)
    b = a[:, :, :3].copy()
    c = a[:, :, -1].copy()
    b[c == 0, :] = 255
    im1 = Image.fromarray(b)
    im1.save(root + '{:03d}.jpg'.format(i))
    #os.system('mv {} {}'.format(root + '{:03d.}'))