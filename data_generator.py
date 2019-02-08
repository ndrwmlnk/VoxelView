import pickle, random, gzip
import numpy as np

class DataGenerator(object):

    def __init__(self, data_file_path):
        """Loads the dataset, randomly selects 16x16x16 voxel patches;
        randomly places these patches into 64x64x64 voxel full spaces;
        returns a batch of BATCHSIZEx64x64x64 training data
        """


        self.voxel_len = 64
        self.voxel_full_empty = np.full((self.voxel_len, self.voxel_len, self.voxel_len), False, dtype=bool)

        with gzip.GzipFile(data_file_path, 'r') as f:
            self.data = pickle.load(f)
        print('>>>  load done  >>>', data_file_path,'\n')

        self.voxelSpaceSize = self.data['voxelSpaceSize']

        self.params_list = []
        self.params_dict = dict()
        for shape in self.data['voxel_training_data'].keys():
            if shape not in self.params_dict.keys():
                self.params_dict[shape] = []
            for size in self.data['voxel_training_data'][shape].keys():
                if type(size) is int:
                    self.params_list.append([shape, size])
                    self.params_dict[shape].append(size)
                    print(shape, size)
                else:
                    print(shape, 'len  >>> ', self.data['voxel_training_data'][shape][size], '\n')  # print len

    def get_voxel_data(self):
        shape = random.choice(list(self.params_dict))
        size = random.choice(self.params_dict[shape])
        r = random.randint(0, len(self.data['voxel_training_data'][shape][size]['voxel'])-1)
        voxel_patch = self.data['voxel_training_data'][shape][size]['voxel'][r]
        return voxel_patch

    def insert_voxel_patch(self, voxel_patch):
        x, y, z = random.randint(0, self.voxel_len-self.voxelSpaceSize), random.randint(0, self.voxel_len-self.voxelSpaceSize), random.randint(0, self.voxel_len-self.voxelSpaceSize)
        voxel_full = self.voxel_full_empty.copy()
        voxel_full[x:x+self.voxelSpaceSize, y:y+self.voxelSpaceSize, z:z+self.voxelSpaceSize] = voxel_patch
        return voxel_full


data_file_path = 'assets/voxel_cube_sphere_pen_233829.pgz'
DG = DataGenerator(data_file_path)

for t in range(100):
    voxel_batch = []
    for b in range(32):
        v_patch = DG.get_voxel_data()  # get a random 16x16x16 voxel_patch from the dataset
        v_full = DG.insert_voxel_patch(v_patch)  # place the 16x16x16 voxel_patch randomly in the 64x64x64 voxel_full space
        voxel = np.array(v_full, dtype='int8').tolist()  # training sample  # dtype='float32'
        voxel_batch.append(voxel)  # training batch
    # VAE.train(voxel_batch)

print('TRAINING DONE')
