import pickle, random, gzip, os
import numpy as np

class DataGenerator(object):

    def __init__(self, data_folder_path):
        """Loads the dataset (episodes), collect a batch of random episode/step samples"""

        lst = os.listdir(data_folder_path)
        lst.sort()
        self.data = {'touch': [], 'hand_proprio': [], 'hand_proprio_vel': [], 'object_pos': [], 'object_pos_vel': [], 'object_quat': [], 'object_quat_vel': [], 'act': [], 'object_goal_pos': [], 'object_goal_quat': []}
        self.eps = 0
        for ll in lst:
            if ll[-4:] == '.pgz':
                fileName = ll
                with gzip.GzipFile(data_folder_path + fileName, 'r') as f:
                    self.ep = pickle.load(f)
                print('>>  load ', fileName)

                self.data['touch'].append(self.ep['o'][0][:, 54:146])
                self.data['hand_proprio'].append(self.ep['o'][0][:, :24])
                self.data['hand_proprio_vel'].append(self.ep['o'][0][:, 24:48])
                self.data['object_pos'].append(self.ep['o'][0][:, 146:149])
                self.data['object_pos_vel'].append(self.ep['o'][0][:, 48:51])
                self.data['object_quat'].append(self.ep['o'][0][:, 149:153])
                self.data['object_quat_vel'].append(self.ep['o'][0][:, 51:54])
                self.data['act'].append(self.ep['u'][0])
                self.data['object_goal_pos'].append(self.ep['g'][0][:, :3])
                self.data['object_goal_quat'].append(self.ep['g'][0][:, 3:])

                self.eps += 1
                self.ep_len = len(self.data['touch'][0])-1

                # from time import gmtime, strftime
                # print(np.array(obs).shape)
                # np.savetxt('obs_ep_' + strftime("%H%M%S", gmtime()) + '.csv', np.array(obs)[:,0,:], delimiter=',', fmt='%.2f')
                # np.savetxt('assets/test.csv', self.ep['o'][:,3,:], delimiter=',', fmt='%.2f')
                # self.data.append({'touch': touch, 'hand_proprio': hand_proprio, 'hand_proprio_vel': hand_proprio_vel, 'object_pos': object_pos, 'object_pos_vel': object_pos_vel, 'object_quat': object_quat, 'object_quat_vel': object_quat_vel, 'act': act, 'object_goal_pos': object_goal_pos, 'object_goal_quat': object_goal_quat})

    def get_voxel_data(self):
        r_ep = random.randint(0, self.eps-1)
        r_step = random.randint(0, self.ep_len-1)

        touch = self.data['touch'][r_ep][r_step]
        hand_proprio = self.data['hand_proprio'][r_ep][r_step]
        object_pos = self.data['object_pos'][r_ep][r_step]
        object_quat = self.data['object_quat'][r_ep][r_step]
        act = self.data['act'][r_ep][r_step]

        return touch, hand_proprio, object_pos, object_quat, act

data_folder_path = "assets/HandManipulateBlockTouchSensors-v0/"  # os.path.expanduser("~")
DG = DataGenerator(data_folder_path)

for t in range(100):
    touch_batch, hand_proprio_batch, object_pos_batch, object_quat_batch, act_batch = [],[],[],[],[]
    for b in range(32):  # collect a batch of random episode/step samples
        touch, hand_proprio, object_pos, object_quat, act = DG.get_voxel_data()  # get a sample of vars from a random episode and a random step

        touch_batch.append(touch)
        hand_proprio_batch.append(hand_proprio)
        object_pos_batch.append(object_pos)
        object_quat_batch.append(object_quat)
        act_batch.append(act)

    # VAE.train([touch_batch, hand_proprio_batch, object_pos_batch, object_quat_batch, act_batch])

print('TRAINING DONE')
