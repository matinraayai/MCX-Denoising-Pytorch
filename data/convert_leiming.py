import os
import scipy.io as spio

path_5 = './checkpoint/square_1e5.mat'
path_6 = './checkpoint/square_1e6.mat'
path_7 = './checkpoint/square_1e7.mat'
path_8 = './checkpoint/square_1e8.mat'
num_tests = 1

for i in range(1, num_tests + 1):
    mat_5 = spio.loadmat(path_5, squeeze_me=True)
    mat_5 = mat_5['currentImage']
    mat_6 = spio.loadmat(path_6, squeeze_me=True)
    mat_6 = mat_6['currentImage']
    mat_7 = spio.loadmat(path_7, squeeze_me=True)
    mat_7 = mat_7['currentImage']
    mat_8 = spio.loadmat(path_8, squeeze_me=True)
    mat_8 = mat_8['currentImage']
spio.savemat('./data/test_snr/square/1.mat', {'x1e5': mat_5, 'x1e6': mat_6, 'x1e7': mat_7, 'x1e8': mat_8})