import os
import numpy as np
import shutil
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cv2 as cv
from scipy.special import expit
from utils import *


conv_layer = [0, 4, 8, 11, 14, 18, 21, 24, 28, 31, 34, 37, 40, 44, 47, 50, 53, 56, 59, 62, 66, 71, 74]
layer_num = 0
def init_weights(m):
    global layer_num
    dir = os.path.join("/", "home", "itlab_sparse_mini", "my_yolo-pytorch", "weights")
    extension = ".csv"
    if type(m) == nn.Conv2d:
        shape = m.weight.shape
        weights = "convWeights"
        path = os.path.join(dir, weights + "_" + str(conv_layer[layer_num]) + "_" + extension)
        print(path)
        data = np.genfromtxt(path, delimiter= " ")
        data = data.reshape(shape)
        m.weight.data = torch.from_numpy(data).float()
        if layer_num == conv_layer[len(conv_layer) - 1]:
            bias_prefix = "biasData"
            bias_path = os.path.join(dir, bias_prefix + "_" + str(conv_layer[layer_num]) + "_" + extension)
            bias = np.genfromtxt(bias_path, delimiter= " ")
            bias = bias.reshape(bias_shape)
            m.bias.data = torch.from_numpy(bias).float()
    elif type(m) == nn.BatchNorm2d:
        bias_shape = m.bias.shape
        weight_shape = m.weight.shape
        weights = "weightsData"
        bias_prefix = "biasData"
        bias_path = os.path.join(dir, bias_prefix + "_" + str(conv_layer[layer_num] + 1) + "_" + extension)
        weight_path = os.path.join(dir, weights + "_" + str(conv_layer[layer_num] + 1) + "_" + extension)
        print(bias_path)
        print(weight_path)

        weight = np.genfromtxt(weight_path, delimiter= " ")
        weight = weight.reshape(weight_shape)
        m.weight.data = torch.from_numpy(weight).float()

        bias = np.genfromtxt(bias_path, delimiter= " ")
        bias = bias.reshape(bias_shape)
        m.bias.data = torch.from_numpy(bias).float()

        if m.track_running_stats:
            running_mean_shape = m.running_mean.shape
            mean_prefix = "meanData"
            mean_path = os.path.join(dir, mean_prefix + "_" + str(conv_layer[layer_num] + 1) + "_" + extension)
            running_mean = np.genfromtxt(mean_path, delimiter= " ")
            running_mean = running_mean.reshape(running_mean_shape)
            m.running_mean.data = torch.from_numpy(running_mean).float()

            running_var_shape = m.running_var.shape
            var_prefix = "stdData"
            var_path = os.path.join(dir, var_prefix + "_" + str(conv_layer[layer_num] + 1) + "_" + extension)
            running_var = np.genfromtxt(var_path, delimiter= " ")
            running_var = running_var.reshape(running_var_shape)
            m.running_var.data = torch.from_numpy(running_var).float()
        layer_num += 1

def run_net(input):
    tensor = torch.from_numpy(input)
    net = Yolov2Voc()
    net2 = Region()
    net.apply(init_weights)
    torch.save(net.state_dict(), "yolov2_voc_model")
    net.eval()

    first_stage = net(tensor)
    first_stage = first_stage.detach().numpy()
    first_stage = first_stage.transpose(0, 2, 3, 1)
    output = net2(first_stage)
    return output

def test(path):

    net = Yolov2Voc()
    net.load_state_dict(torch.load("yolov2_voc_model"))
    net.eval()
    net2 = Region()
    net2.eval()
    for image_path in glob.glob(os.path.join(path, '*.jpg')):
        path_list = image_path.split('/')
        name = path_list[-1].split('.')[0] + ".txt"
        image = cv.imread(image_path)
        input = preprocessing(image)  
        tensor = torch.from_numpy(input)
        first_stage = net(tensor)
        first_stage = first_stage.detach().numpy()
        first_stage = first_stage.transpose(0, 2, 3, 1)
        output = net2(first_stage)
        postprocess(image, output, write=True, filename=name)
