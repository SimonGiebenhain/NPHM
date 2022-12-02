import trimesh
import os

root = '/Users/sigi/phd/test_project_page2/docs/models/'

files = [f for f in os.listdir(root) if 'mesh' in f and f.endswith('ply')]

for f in files:
    m = trimesh.load(root + f)
    m = m.simplify_quadratic_decimation(100000)
    m.export(root + f)
