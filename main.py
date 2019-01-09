import numpy as np
from pyquaternion import Quaternion
import pickle
from time import gmtime, strftime

def generate_cloud(size=11,dimension=0.025):
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
    return voxel_range / max_dimension


def normaliza(cube_points, voxel_range):
    # normalizing the shape points to always touch limits of a voxel space
    norm_cube_points = np.multiply(cube_points, find_scaling(cube_points, voxel_range))
    return norm_cube_points


def cloud2voxel(cloud, voxel_range, size=11):
    # convert to voxel grid
    epsilon = 0.1
    size05 = size / 2
    voxels = np.zeros((size, size, size), dtype=np.uint8)
    pos_center = (size05, size05, size05)
    for pt in cloud:
        i, j, k = np.floor((pt - epsilon) * size05 / voxel_range + pos_center)
        voxels[int(i)][int(j)][int(k)] = 1
    return voxels


if __name__ == "__main__":
    voxelSpaceSize = 11
    voxelRange = 2.0
    dimension = 0.025
    cloud = generate_cloud(size=voxelSpaceSize, dimension=dimension)
    numberOfShapes = 10  # 10000
    seedid = 0

    quaternions = []
    cloudShapes = []
    voxelShapes = []
    np.random.seed(seedid)

    for i in range(numberOfShapes):
        quat = Quaternion.random()
        if i == 0: quat = Quaternion()  # first quat = 1 + 0i + 0j + 0k
        if i % 100 == 0: print(strftime("%H%M%S", gmtime()), '\t', i, '\t', quat)
        cloud_transformed = normaliza(transform_cloud(cloud, quat), voxelRange)
        voxels = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
        # gets all surface voxels marked with a 1 -- used for checking
        surfaceVoxels = voxels[0, :, :].ravel().sum() + voxels[:, 0, :].ravel().sum() + voxels[:, :, 0].ravel().sum() + voxels[-1, :, :].ravel().sum() + voxels[:, -1, :].ravel().sum() + voxels[:, :, -1].ravel().sum()
        assert(0 < surfaceVoxels)
        quaternions.append(quat)
        cloudShapes.append(cloud_transformed)
        voxelShapes.append(voxels)

    with open('voxel_shapes_len' + str(len(voxelShapes)) + '_' + strftime("%H%M%S", gmtime()) + '.pkl', 'wb') as f:
        pickle.dump({'quaternions': quaternions, 'cloudShapes': cloudShapes, 'voxelShapes': voxelShapes, 'voxelSpaceSize': voxelSpaceSize, 'voxelRange': voxelRange, 'dimension': dimension, 'cloud': cloud, 'numberOfShapes': numberOfShapes, 'seedid': seedid}, f)

    print('DONE')
