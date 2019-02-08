import pickle, os, gzip
import numpy as np
from pyquaternion import Quaternion
from time import gmtime, strftime
import math
import matplotlib

matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def generate_cloud_cube(size=16,dimension=1.0):
    # generates point cloud with center 0,0,0
    pts = []
    for i in np.linspace(-dimension, dimension, size):
        for j in np.linspace(-dimension, dimension, size):
            for k in np.linspace(-dimension, dimension, size):
                if abs(i) == dimension or abs(j) == dimension or abs(k) == dimension:
                    pts.append([i, j, k])
    return np.array(pts, dtype='float32')


def generate_cloud_sphere(size=16,dimension=1.0):
    # samples = int(round(size * size * np.sqrt(size)))
    samples = int(round(size * size * size))
    pts = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))
        phi = ((i + 1.) % samples) * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        pts.append([x * dimension, y * dimension, z * dimension])
    return np.array(pts, dtype='float32')


def generate_cloud_pen(size=16, dimension=1.0):
    pts = []
    pen_2r = 0.008
    pen_len = 0.1
    r = (pen_2r / pen_len * dimension) / 2
    # samples_r = np.linspace(0, 2, int(round(size/2)))
    samples_r = np.linspace(0, 2, 16+1)
    x, y = r * np.cos(samples_r * np.pi)[:-1], r * np.sin(samples_r * np.pi)[:-1]
    l = np.linspace(-dimension, dimension, size)
    for i in range(len(x)):
        for zz in l:
            pts.append([x[i], y[i], zz])
    pts.append([0.0, 0.0, -dimension - r])
    pts.append([0.0, 0.0, dimension + r])
    return np.array(pts, dtype='float32')


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


def cloud2voxel(cloud, voxel_range, size=16, shift=[0, 0, 0]):
    # convert to voxel grid
    size05 = (size - 1) / 2
    voxels = np.zeros((size, size, size), dtype=bool)
    pos_center = (size05, size05, size05)
    for pt in cloud:
        i, j, k = np.round(pt * size05 / voxel_range + pos_center).astype('int')
        # print(pt * size05 / voxel_range + pos_center, i, j, k)  # debugging
        i += shift[0]
        j += shift[1]
        k += shift[2]
        if i < size and j < size and k < size and i >= 0 and j >= 0 and k >= 0:
            voxels[int(i)][int(j)][int(k)] = 1
        else:
            print('quit()  >>  ', i, j, k)
            return None
            # quit()
    return voxels


if __name__ == "__main__":
    trg_gen_type = 'generate_vae_training_data'  # rotate_q111 generate_vae_training_data move_q0 load_quat_seq_ep.pgz
    seedid = 42

    np.random.seed(seedid)

    if trg_gen_type == 'rotate_q111':
        trg_normalize_cloud_size = True
        numberOfShapes = 36  # 10000

        # rotation of the points initialized in the 3D-voxel-cube may bring some of these points (e.g. vertices) outside of the 3D-voxel-cube
        # to ensure all points are in the 3D-voxel-cube after a rotation, either set trg_normalize_cloud_size = True or set the ratio dimension/dimension <= ~0.6
        voxelSpaceSize = 16
        voxelRange = 2.0
        dimension = 1.0
        cloud = generate_cloud_cube(size=voxelSpaceSize, dimension=dimension)

        quaternions = []
        cloudShapes = []
        voxelShapes = []

        for i in range(numberOfShapes):
            quat = Quaternion.random()

            angles = np.linspace(0, 2, numberOfShapes)
            quat = Quaternion(axis=[1, 1, 1], angle=np.pi * angles[i])
            if i == 0: quat = Quaternion(axis=[1, 1, 1], angle=np.pi / 4)
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

        with gzip.GzipFile('assets' + os.sep + 'voxel_shapes_len' + str(len(voxelShapes)) + '_' + strftime("%H%M%S", gmtime()) + '.pgz', 'w') as f:
            # with open('assets' + os.sep + 'voxel_shapes_len' + str(len(voxelShapes)) + '_' + strftime("%H%M%S", gmtime()) + '.pkl', 'wb') as f:
            pickle.dump({'quaternions': quaternions, 'cloudShapes': cloudShapes, 'voxelShapes': voxelShapes, 'voxelSpaceSize': voxelSpaceSize, 'voxelRange': voxelRange, 'dimension': dimension, 'cloud': cloud, 'seedid': seedid}, f)

    elif trg_gen_type == 'move_q0':
        trg_normalize_cloud_size = False

        # rotation of the points initialized in the 3D-voxel-cube may bring some of these points (e.g. vertices) outside of the 3D-voxel-cube
        # to ensure all points are in the 3D-voxel-cube after a rotation, either set trg_normalize_cloud_size = True or set the ratio dimension/dimension <= ~0.6
        voxelSpaceSize = 16
        voxelRange = 2.0
        dimension = 1.0
        cloud = generate_cloud_cube(size=voxelSpaceSize, dimension=dimension)

        quaternions = []
        cloudShapes = []
        voxelShapes = []

        quat = Quaternion(axis=[1, 1, 1], angle=0)

        a = [range(-3, 4), range(-3, 4), range(-3, 4)]
        # a = [range(-4, 5), range(-4, 5), range(-4, 5)]
        import itertools
        shifts3D = list(itertools.product(*a))

        for i in range(len(shifts3D)):
            if i % 100 == 0: print(strftime("%H%M%S", gmtime()), '\t', i, '\t', quat)
            cloud_transformed = cloud
            voxels = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize, shift=shifts3D[i])
            quaternions.append(quat)
            cloudShapes.append(cloud_transformed)
            voxelShapes.append(voxels)

        with gzip.GzipFile('assets' + os.sep + 'voxel_shapes_len' + str(len(voxelShapes)) + '_' + strftime("%H%M%S", gmtime()) + '.pgz', 'w') as f:
            # with open('assets' + os.sep + 'voxel_shapes_len' + str(len(voxelShapes)) + '_' + strftime("%H%M%S", gmtime()) + '.pkl', 'wb') as f:
            pickle.dump({'quaternions': quaternions, 'cloudShapes': cloudShapes, 'voxelShapes': voxelShapes, 'voxelSpaceSize': voxelSpaceSize, 'voxelRange': voxelRange, 'dimension': dimension, 'cloud': cloud, 'seedid': seedid}, f)

    elif trg_gen_type == 'load_quat_seq_ep.pgz':
        trg_normalize_cloud_size = False

        # rotation of the points initialized in the 3D-voxel-cube may bring some of these points (e.g. vertices) outside of the 3D-voxel-cube
        # to ensure all points are in the 3D-voxel-cube after a rotation, either set trg_normalize_cloud_size = True or set the ratio dimension/dimension <= ~0.6
        voxelSpaceSize = 16
        voxelRange = 2.0
        dimension = 1.0
        cloud = generate_cloud_cube(size=voxelSpaceSize, dimension=dimension)

        quaternions = []
        cloudShapes = []
        voxelShapes = []

        file = 'tmp/0118_101917/ep.pgz'
        with gzip.open(file, 'r') as f:
            data = pickle.load(f)
        for k in data.keys():
            print(k, type(data[k]))
        for i in range(len(data['achieved_goals'])):
            quat = Quaternion(data['achieved_goals'][i][0][3:])
            if i % 100 == 0: print(strftime("%H%M%S", gmtime()), '\t', i, '\t', quat)
            if trg_normalize_cloud_size:
                cloud_transformed = normaliza(transform_cloud(cloud, quat), voxelRange)
            else:
                cloud_transformed = transform_cloud(cloud, quat)
            voxels = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
            quaternions.append(quat)
            cloudShapes.append(cloud_transformed)
            voxelShapes.append(voxels)

        with gzip.GzipFile('assets' + os.sep + 'voxel_shapes_len' + str(len(voxelShapes)) + '_' + strftime("%H%M%S", gmtime()) + '.pgz', 'w') as f:
            # with open('assets' + os.sep + 'voxel_shapes_len' + str(len(voxelShapes)) + '_' + strftime("%H%M%S", gmtime()) + '.pkl', 'wb') as f:
            pickle.dump({'quaternions': quaternions, 'cloudShapes': cloudShapes, 'voxelShapes': voxelShapes, 'voxelSpaceSize': voxelSpaceSize, 'voxelRange': voxelRange, 'dimension': dimension, 'cloud': cloud, 'seedid': seedid}, f)

    elif trg_gen_type == 'generate_vae_training_data':
        voxel_training_data = dict()
        voxel_training_data['cube'] = {}
        voxel_training_data['pen'] = {}
        voxel_training_data['sphere'] = {}

        # rotation of the points initialized in the 3D-voxel-cube may bring some of these points (e.g. vertices) outside of the 3D-voxel-cube
        # to ensure all points are in the 3D-voxel-cube after a rotation, either set trg_normalize_cloud_size = True or set the ratio dimension/dimension <= ~0.6
        voxelSpaceSize = 16
        voxelRange = 2.0

        range_quat_cube = 30  #* 1000
        range_quat_pen = 10  #*1000


        l = 0
        for shapeSize in [8, 9, 10, 11]:
            if shapeSize not in voxel_training_data['cube'].keys():
                voxel_training_data['cube'][shapeSize] = {}
                voxel_training_data['cube'][shapeSize]['quat'] = []
                voxel_training_data['cube'][shapeSize]['voxel'] = []
                # voxel_training_data['cube']['cloud'] = []
            dimension = voxelRange * shapeSize / voxelSpaceSize
            cloud = generate_cloud_cube(size=voxelSpaceSize, dimension=dimension)
            for i in range(range_quat_cube):
                quat = Quaternion.random()
                # if i == 0: quat = Quaternion(axis=[1, 1, 1], angle=np.pi / 4)
                if i == 0: quat = Quaternion()  # first quat = 1 + 0i + 0j + 0k
                if i % 100 == 0: print(strftime("%H%M%S", gmtime()), 'cube', '\t', 'shapeSize', shapeSize, '\t', 'i', i, '\t', 'quat', quat)
                cloud_transformed = transform_cloud(cloud, quat)
                voxels = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
                if voxels is not None:
                    voxel_training_data['cube'][shapeSize]['quat'].append(quat)
                    voxel_training_data['cube'][shapeSize]['voxel'].append(voxels)
                    # voxel_training_data['cube'][shapeSize]['cloud_transformed'].append(cloud_transformed)
                    l += 1
        voxel_training_data['cube']['_len'] = l
        print(l, ' cube voxels generated')

        save_name = 'assets' + os.sep + 'voxel_cube_' + strftime("%H%M%S", gmtime()) + '.pgz'
        print('>>>  saving data  >>>  ',save_name)
        with gzip.GzipFile(save_name, 'w') as f:
            pickle.dump({'voxel_training_data': voxel_training_data, 'voxelSpaceSize': voxelSpaceSize, 'seedid': seedid}, f)


        l = 0
        for shapeSize in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
            if shapeSize not in voxel_training_data['sphere'].keys():
                voxel_training_data['sphere'][shapeSize] = {}
                voxel_training_data['sphere'][shapeSize]['quat'] = []
                voxel_training_data['sphere'][shapeSize]['voxel'] = []
                # voxel_training_data['cube']['cloud'] = []
            dimension = voxelRange * shapeSize / voxelSpaceSize
            cloud = generate_cloud_sphere(size=voxelSpaceSize, dimension=dimension)

            quat = Quaternion()
            cloud_transformed = transform_cloud(cloud, quat)
            print(strftime("%H%M%S", gmtime()), 'sphere', '\t', 'shapeSize', shapeSize, '\t', 'i', i, '\t', 'quat', quat)

            # cloud = np.array(cloud)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlim3d(-1.5, 1.5)
            # ax.set_ylim3d(-1.5, 1.5)
            # ax.set_zlim3d(-1.5, 1.5)
            # ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2])
            # plt.show()

            voxels = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
            voxel_training_data['sphere'][shapeSize]['quat'].append(quat)
            voxel_training_data['sphere'][shapeSize]['voxel'].append(voxels)
            # voxel_training_data['sphere'][shapeSize]['cloud_transformed'].append(cloud_transformed)
            l += 1
        voxel_training_data['sphere']['_len'] = l
        print(l, ' sphere voxels generated')

        save_name = 'assets' + os.sep + 'voxel_cube_sphere_' + strftime("%H%M%S", gmtime()) + '.pgz'
        print('>>>  saving data  >>>  ',save_name)
        with gzip.GzipFile(save_name, 'w') as f:
            pickle.dump({'voxel_training_data': voxel_training_data, 'voxelSpaceSize': voxelSpaceSize, 'seedid': seedid}, f)


        l = 0
        for shapeSize in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
            if shapeSize not in voxel_training_data['pen'].keys():
                voxel_training_data['pen'][shapeSize] = {}
                voxel_training_data['pen'][shapeSize]['quat'] = []
                voxel_training_data['pen'][shapeSize]['voxel'] = []
                # voxel_training_data['pen']['cloud'] = []
            dimension = voxelRange * shapeSize / voxelSpaceSize
            cloud = generate_cloud_pen(size=voxelSpaceSize, dimension=dimension)
            for i in range(range_quat_pen):
                quat = Quaternion.random()
                # if i == 0: quat = Quaternion(axis=[1, 1, 1], angle=np.pi / 4)
                if i == 0: quat = Quaternion()  # first quat = 1 + 0i + 0j + 0k
                if i % 100 == 0: print(strftime("%H%M%S", gmtime()), 'pen', '\t', 'shapeSize', shapeSize, '\t', 'i', i, '\t', 'quat', quat)
                cloud_transformed = transform_cloud(cloud, quat)
                voxels = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
                if voxels is not None:
                    voxel_training_data['pen'][shapeSize]['quat'].append(quat)
                    voxel_training_data['pen'][shapeSize]['voxel'].append(voxels)
                    # voxel_training_data['pen'][shapeSize]['cloud_transformed'].append(cloud_transformed)
                    l += 1
        voxel_training_data['pen']['_len'] = l
        print(l, ' pen voxels generated')

        save_name = 'assets' + os.sep + 'voxel_cube_sphere_pen_' + strftime("%H%M%S", gmtime()) + '.pgz'
        print('>>>  saving data  >>>  ',save_name)
        with gzip.GzipFile(save_name, 'w') as f:
            pickle.dump({'voxel_training_data': voxel_training_data, 'voxelSpaceSize': voxelSpaceSize, 'seedid': seedid}, f)

    print('DONE')
