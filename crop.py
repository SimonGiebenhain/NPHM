import os
from PIL import Image
import numpy as np

root = '/Users/sigi/phd/test_project_page2/docs/static/latent_interpolation/chair/'

files = [f for f in os.listdir(root) if f.endswith('.jpg')]

files.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
print(files)


for i, f in enumerate(files):
    im1 = Image.open(root + f)
    a = np.array(im1)
    im1 = Image.fromarray(a[:, 128:-128, :])
    im1.save(root + '{:03d}.jpg'.format(i))
    #os.system('mv {} {}'.format(root + '{:03d.}'))