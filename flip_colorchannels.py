from PIL import Image
import numpy as np
import os

root = '/Users/sigi/phd/test_project_page2/docs/static/figures/shape/'

files = [f for f in os.listdir(root) if f.endswith('.png')]

for f in files:
    I = np.array(Image.open(root + f))
    I = I[:, :, (2, 1, 0, 3)]
    I = Image.fromarray(I)
    I.show()
    I.save(root + f)