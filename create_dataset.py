import multiprocessing
import pickle, os, gzip
import sys
from collections import deque

import numpy as np
from pyquaternion import Quaternion
import time
import math


class Generator:

    @staticmethod
    def generate_cloud_cube(size=16, scale=1.0):
        # generates point cloud with center 0,0,0
        pts = []
        for i in np.linspace(-scale, scale, size):
            for j in np.linspace(-scale, scale, size):
                for k in np.linspace(-scale, scale, size):
                    if abs(i) == scale or abs(j) == scale or abs(k) == scale:
                        pts.append([i, j, k])
        return np.array(pts, dtype='float32')

    @staticmethod
    def generate_cloud_sphere(size=16, scale=1.0):
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
            pts.append([x * scale, y * scale, z * scale])
        return np.array(pts, dtype='float32')

    @staticmethod
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

    @staticmethod
    def transform_cloud(point_cloud, quaternion, cube_position=[0.0, 0.0, 0.0]):
        pts_transformed = []
        for pt in point_cloud:
            pts_transformed.append(quaternion.rotate(pt) + cube_position)
        return pts_transformed

    @staticmethod
    def cloud2voxel(cloud, voxel_range, size, shift=[0, 0, 0]):
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
            if 0 <= i < size and 0 <= j < size and 0 <= k < size:
                voxels[int(i)][int(j)][int(k)] = 1
            else:
                # print('quit()  >>  ', i, j, k, voxel_range, shift)
                return None
                # quit()
        return voxels

    @staticmethod
    def gen_and_transform(shape, space_size, size, rotation, shift=None):

        def calc_shift(perc):
            return (space_size-size)*perc - (size/2)

        shift = [calc_shift(x) for x in shift] if shift is not None else [0.5, 0.5, 0.5]
        scale = 2.0 * float(size) / float(space_size)
        generate_cloud = getattr(Generator, 'generate_cloud_%s' % shape)
        cloud_transformed = Generator.transform_cloud(generate_cloud(size=space_size, scale=scale),
                                                      rotation)
        return Generator.cloud2voxel(cloud_transformed, 2.0, size=space_size, shift=shift)

    @staticmethod
    def generate_shape_variations(shape, space_size, params):
        # get rotations and sizes from params or set defaults

        def rnd_quat(rot_sample):
            quat = Quaternion()
            axis_lookup = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
            for axis, increment_degrees in rot_sample.items():
                angle = np.random.choice(np.arange(0, 2*math.pi, increment_degrees*math.pi/180)) if \
                    0 < increment_degrees < 360 else np.random.rand()*2*math.pi
                quat *= Quaternion(axis=axis_lookup[axis], angle=angle)

            return quat

        def rnd_shift(shift_steps):
            shift = []
            for axis in ['x', 'y', 'z']:
                try:
                    inc = max(min(shift_steps[axis]/100, 1), 0)
                    if inc > 0:
                        perc = np.random.choice(np.linspace(0, 1, int(1.0/inc)))
                    else:
                        perc = 0
                except KeyError:
                    perc = 0.5
                shift.append(perc)
            return shift

        rotations = params.get('rotation_steps', {})
        sizes = params.get('sizes', [space_size/2])
        shift_steps = params.get('shift_step_perc', {})

        # setup job-configs

        num_items = params.get('amount', 100)
        args = [(shape, space_size, sizes[np.random.choice(range(len(sizes)))],
                 rnd_quat(rotations),
                 rnd_shift(shift_steps))
                for _ in range(num_items)]

        print('trying to generate', num_items, '%ss' % shape)
        # initialize worker pool
        pool = multiprocessing.Pool(processes=min(num_items, multiprocessing.cpu_count()))

        result = deque()
        done = deque()
        regen = deque()

        def update(res):
            if res is not None:
                done.append(1)
                result.append(res)
                sys.stdout.write('\r%r/%r [%r failed] => %r %% done' % (len(done), len(args), len(regen),
                                                            round(100*len(done)/len(args), 2)))
            else:
                regen.append(1)
                pool.apply_async(Generator.gen_and_transform, (shape, space_size, sizes[np.random.choice(range(len(sizes)))],
                 rnd_quat(rotations),
                 rnd_shift(shift_steps)), callback=update)

        for arg in args:
            pool.apply_async(Generator.gen_and_transform, arg, callback=update)
        while not len(done) == len(args):
            time.sleep(0.1)
        pool.close()
        pool.join()
        print('\r100% done!')

        return result

    @staticmethod
    def create_dataset(space_size, params, file_name='{shapes}_{total}', directory=None):

        abs_dir = os.path.dirname(os.path.abspath(__file__))
        if directory is None:
            directory = abs_dir

        if os.path.isabs(directory):
            file_name = '{}{}{}'.format(directory, os.sep, file_name)
        else:
            file_name = '{}{}{}{}'.format(abs_dir, os.sep, directory, os.sep, file_name)

        gen_data = []
        for shape in params.keys():
            start = time.time()
            shape_data = Generator.generate_shape_variations(shape, space_size, params[shape])
            gen_data.extend(shape_data)
            print('generated %r %ss in %r s' % (len(shape_data), shape, round(time.time() - start, 2)))

        name_dict = {'shapes': '_'.join(params.keys()),
                     'total': len(gen_data)}
        file_name = file_name.format(**name_dict)
        np.save(file_name, gen_data)
        print('saved %r shapes to %s.npy' % (len(gen_data), file_name))


def create_example_dataset():
    space_size = 16
    params = {
        'cube': {'sizes': range(8, 9),
                 'rotation_steps': {'x': 45, 'y': 45, 'z': 45},
                 'shift_step_perc': {'x': 20, 'y': 20, 'z': 20},
                 'amount': 100
                 },
        'sphere': {'sizes': range(5, 11),
                   'amount': 100}
    }
    Generator.create_dataset(space_size, params)


if __name__ == "__main__":
    create_example_dataset()
