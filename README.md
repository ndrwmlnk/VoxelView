# VoxelView
The function performs rotation of a voxelized representation in accordance with a given quaternion.

![quat_111.mp4](https://raw.githubusercontent.com/ndrwmlnk/VoxelView/master/assets/quat_111.gif)

# Read '*.pgz' files:

    import gzip
    file = 'assets/voxel_shapes_len729_ss16_cs8_q0_shift4px.pgz'
    file = 'assets/voxel_shapes_len343_ss16_cs10_q0_shift3px.pgz'
    with gzip.open(file, 'r') as f:
        data = pickle.load(f)
    for k in data.keys():
        print(k, type(data[k]))
        

# voxel_shapes_len343_ss16_cs10_q0_shift3px

len343 - 343 saved voxel representations\
ss16 - 16x16x16 voxel space\
cs10 - cube size 10x10x10 voxels\
q0 - quaternion 1+0i+0j+0k\
shift3px - shifting the cube [-3, -2, -1, 0, 1, 2, 3] at all axes 
    