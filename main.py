import pickle, os
import numpy as np
from pyquaternion import Quaternion
from time import gmtime, strftime

def generate_cloud(size=11,dimension=1.0):
    # generates point cloud with center 0,0,0
    pts = []
    for i in np.linspace(-dimension, dimension, size):
        for j in np.linspace(-dimension, dimension, size):
            for k in np.linspace(-dimension, dimension, size):
                if abs(i) == dimension or abs(j) == dimension or abs(k) == dimension:
                    pts.append(np.array([i, j, k]))
    return pts


def transform_cloud(point_cloud, quaternion, cube_position=[0.0, 0.0, 0.0]):
    pts_transformed = []
    for pt in point_cloud:
        pts_transformed.append(quaternion.rotate(pt) + cube_position)
    return pts_transformed


def find_scaling(cloud_transformed, voxel_range):
    max_dimension = max(list(map(lambda x: max(x), cloud_transformed)))
    # print("find_scaling: return = ", voxel_range / max_dimension)
    return voxel_range / max_dimension


def normaliza(cube_points, voxel_range):
    # normalizing the shape points to always touch limits of a voxel space
    norm_cube_points = np.multiply(cube_points, find_scaling(cube_points, voxel_range))
    return norm_cube_points


def cloud2voxel(cloud, voxel_range, size=11):
    # convert to voxel grid
    size05 = (size - 1) / 2
    voxels = np.zeros((size, size, size), dtype=np.uint8)
    pos_center = (size05, size05, size05)
    for pt in cloud:
        i, j, k = np.round(pt * size05 / voxel_range + pos_center).astype('int')
        # print(pt * size05 / voxel_range + pos_center, i, j, k)  # debugging
        # if i < size and j < size and k < size:
        voxels[int(i)][int(j)][int(k)] = 1
    return voxels


if __name__ == "__main__":
    voxelSpaceSize = 16
    voxelRange = 2.0
    dimension = 1.25
    trg_normalize_cloud_size = False
    # rotation of the points initialized in the 3D-voxel-cube may bring some of these points (e.g. vertices) outside of the 3D-voxel-cube
    # to ensure all points are in the 3D-voxel-cube after a rotation, either set trg_normalize_cloud_size = True or set the ratio dimension/dimension <= ~0.6
    cloud = generate_cloud(size=voxelSpaceSize, dimension=dimension)
    numberOfShapes = 36  # 10000
    seedid = 0

    quaternions = []
    cloudShapes = []
    voxelShapes = []
    np.random.seed(seedid)

    for i in range(numberOfShapes):
        quat = Quaternion.random()

        angles = np.linspace(0, 2, numberOfShapes)
        quat = Quaternion(axis=[1, 1, 1], angle=np.pi * angles[i])
        # if i == 0: quat = Quaternion(axis=[1, 1, 1], angle=np.pi / 4)
        # if i == 0: quat = Quaternion()  # first quat = 1 + 0i + 0j + 0k

        if i % 100 == 0: print(strftime("%H%M%S", gmtime()), '\t', i, '\t', quat)
        if trg_normalize_cloud_size:
            cloud_transformed = normaliza(transform_cloud(cloud, quat), voxelRange)
        else:
            cloud_transformed = transform_cloud(cloud, quat)
        voxels = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
        # gets all surface voxels marked with a 1 -- used for checking
        surfaceVoxels = voxels[0, :, :].ravel().sum() + voxels[:, 0, :].ravel().sum() + voxels[:, :, 0].ravel().sum() + voxels[-1, :, :].ravel().sum() + voxels[:, -1, :].ravel().sum() + voxels[:, :, -1].ravel().sum()
        assert(0 < surfaceVoxels)
        quaternions.append(quat)
        cloudShapes.append(cloud_transformed)
        voxelShapes.append(voxels)

    with open('assets' + os.sep + 'voxel_shapes_len' + str(len(voxelShapes)) + '_' + strftime("%H%M%S", gmtime()) + '.pkl', 'wb') as f:
        pickle.dump({'quaternions': quaternions, 'cloudShapes': cloudShapes, 'voxelShapes': voxelShapes, 'voxelSpaceSize': voxelSpaceSize, 'voxelRange': voxelRange, 'dimension': dimension, 'cloud': cloud, 'numberOfShapes': numberOfShapes, 'seedid': seedid}, f)

    print('DONE')
