import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import  # This import registers the 3D projection, but is otherwise unused.

file = 'voxel_shapes_len10_000000.pkl'
with open(file, 'rb') as f:
    voxel_shapes = pickle.load(f)

print('\n', 'matplotlib  >>  draw the first voxel_shape from:  ' + file,'\n')
quat = voxel_shapes['quaternions'][0]
voxels = voxel_shapes['voxelShapes'][0]>0
print('quat = ', quat, '\n')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=[1, 0, 0, 0.3], edgecolor='k')
ax.view_init(0, 181)
plt.draw()
plt.pause(.001)

for angle in range(181, 270):
    if angle % 10 == 0: print('view', 0, angle)
    ax.view_init(0, angle)
    plt.draw()
    plt.pause(.001)

plt.pause(100)
print('EXIT')
