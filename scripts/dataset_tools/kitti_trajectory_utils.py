# -*- coding: utf-8 -*- 
# @Time : 2020/9/28 13:07 
# @Author : CaiXin
# @File : kitti_trajectory_utils.py

'''将kitti数据集的位姿转换成toolbox要求的格式'''
import argparse
import math
import os

import numpy as np


def quat2dcm(quaternion):
    """Returns direct cosine matrix from quaternion (Hamiltonian, [x y z w])
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1])),
        dtype=np.float64)


def dcm2quat(matrix_3x3):
    """Return quaternion (Hamiltonian, [x y z w]) from rotation matrix.
    This algorithm comes from  "Quaternion Calculus and Fast Animation",
    Ken Shoemake, 1987 SIGGRAPH course notes
    (from Eigen)
    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix_3x3, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > 0.0:
        t = math.sqrt(t + 1.0)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (M[2, 1] - M[1, 2]) * t
        q[1] = (M[0, 2] - M[2, 0]) * t
        q[2] = (M[1, 0] - M[0, 1]) * t
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = math.sqrt(M[i, i] - M[j, j] - M[k, k] + 1.0)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (M[k, j] - M[j, k]) * t
        q[j] = (M[i, j] + M[j, i]) * t
        q[k] = (M[k, i] + M[i, k]) * t
    return q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Analyze trajectories''')
    parser.add_argument('--data-root', type=str, default=' ', help='Kitti odometry dataset folder')
    parser.add_argument('--est-root', type=str, default=' ', help='Folder that estimated pose file exists')
    parser.add_argument('--sequence-idx', type=str, default='09', help='Specify the sequence to be converted')
    parser.add_argument('--gt', type=str, default='True', help='if false, use estimated poses')
    parser.add_argument('--filename', type=str, default='est_{0}.txt', help='filename of estimated poses, if with "--gt True", ignore this')
    parser.add_argument('--out-dir', type=str, default='../../kitti/', help='to save the trajectory of the modified format')
    args = parser.parse_args()
    assert os.path.exists(args.data_root)

    # read raw timestamps and poses
    data_root = args.data_root
    sequence_name = args.sequence_idx
    is_groundtruth = args.gt
    timestamp_file = os.path.join(data_root, 'sequences', sequence_name, 'times.txt')
    if is_groundtruth == 'True':
        poses_file = os.path.join(data_root, 'poses', '{0}.txt'.format(sequence_name))
    else:
        poses_file = os.path.join(args.est_root, args.filename)

    timestamps = np.genfromtxt(timestamp_file).astype(np.float64)
    poses = np.genfromtxt(poses_file).astype(np.float64)
    transform_matrices = poses.reshape(-1, 3, 4)  # 此时的转换矩阵是3*4的形式

    # result cache
    N = len(timestamps)-1# 此处减1是因为VOLO的输出位姿个数比真值个数少1
    positions = np.zeros([N, 3])
    quats = np.zeros([N, 4])
    for i in range(N):
        positions[i, :] = transform_matrices[i, :, -1].reshape(1, 3)
        quats[i, :] = dcm2quat(transform_matrices[i, :, :3])

    # write the result file
    file_lines = []
    header = '#timestamp tx ty tz qx qy qz qw\n'
    file_lines.append(header)
    for i in range(N):
        file_lines.append(''.join([str('%e' % timestamps[i]), ' ',
                                   str('%e' % positions[i, 0]), ' ',
                                   str('%e' % positions[i, 1]), ' ',
                                   str('%e' % positions[i, 2]), ' ',
                                   str('%e' % quats[i, 0]), ' ',
                                   str('%e' % quats[i, 1]), ' ',
                                   str('%e' % quats[i, 2]), ' ',
                                   str('%e' % quats[i, 3]), '\n']))

    out_dir = os.path.join(args.out_dir, sequence_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if is_groundtruth == 'True':
        outfn = os.path.join(out_dir, 'stamped_groundtruth.txt')
    else:
        outfn = os.path.join(out_dir, 'stamped_traj_estimate.txt')
    with open(outfn, 'w') as f:
        f.writelines(file_lines)
