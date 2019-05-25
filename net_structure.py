import os
import numpy as np
import shutil
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cv2 as cv
from scipy.special import expit


class Yolov2Voc(nn.Module):
    def __init__(self):
        super(Yolov2Voc, self).__init__()
        self.first_stage = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, (3,3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(32, eps=1e-06, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 2
            nn.MaxPool2d((2,2), (2,2)),
            # Layer 3
            nn.Conv2d(32, 64, (3,3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(64, eps=1e-06, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 4
            nn.MaxPool2d((2,2), (2,2), ceil_mode=True),
            # Layer 5
            nn.Conv2d(64, 128, (3,3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(128, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # # Layer 6
            nn.Conv2d(128, 64, (1, 1), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(64, eps=1e-06, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 7
            nn.Conv2d(64, 128, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(128, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 8
            nn.MaxPool2d((2,2), (2,2), ceil_mode=True),
            # Layer 9
            nn.Conv2d(128, 256, (3, 3), (1,1), (1, 1), bias=False),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 10
            nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(128, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 11
            nn.Conv2d(128, 256, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 12
            nn.MaxPool2d((2,2), (2,2)),
            # Layer 13
            nn.Conv2d(256, 512, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(512, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 14
            nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 15
            nn.Conv2d(256, 512, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(512, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 16
            nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 17
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.second_stage_1 = nn.Sequential(
            # Layer 18
            nn.MaxPool2d((2,2), (2,2)),
            # Layer 19
            nn.Conv2d(512, 1024, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(1024, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 20
            nn.Conv2d(1024, 512, (1, 1), (1,1), (0, 0), bias=False),
            nn.BatchNorm2d(512, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 21
            nn.Conv2d(512, 1024, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(1024, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 22
            nn.Conv2d(1024, 512, (1, 1), (1,1), (0, 0), bias=False),
            nn.BatchNorm2d(512, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 23
            nn.Conv2d(512, 1024, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(1024, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 24
            nn.Conv2d(1024, 1024, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(1024, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 25
            nn.Conv2d(1024, 1024, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(1024, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.second_stage_2 = nn.Sequential(
            # Layer 27
            nn.Conv2d(512, 64, (1, 1), (1,1), (0, 0), bias=False),
            nn.BatchNorm2d(64, eps=1e-06, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.second_stage_3 = nn.Sequential(
            # Layer 30
            nn.Conv2d(1280, 1024, (3, 3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(1024, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.1),
            # Layer 31
            nn.Conv2d(1024, 125, (1, 1), (1,1), (0, 0), bias=True)
        )

    def forward(self, input):
        # Layers 1 - 17
        x = self.first_stage(input)
        # Layers 18 - 25
        y = self.second_stage_1(x)
        # Layer 26 - copy out 17 layer (route)
        x_copy = x
        # Layer 27
        x_copy = self.second_stage_2(x_copy)
        # Layer 28 (reorg)
        batch_size, num_channel, height, width = x_copy.data.size()
        output_2 = x_copy.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        x_copy = output_2.view(batch_size, -1, int(height / 2), int(width / 2))
        # Layer 29: route -1 with -4
        xy_cat = torch.cat((x_copy, y), 1)
        # Layer 30-31
        result = self.second_stage_3(xy_cat)
        return result



class Region(nn.Module):
    def __init__(self):
        super(Region, self).__init__()
        self.thresh = 0.5
        self.coords = 4
        self.classes = 20
        self.anchors = 5 # num
        self.bias =  torch.tensor([1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071])
        self.nmsthresh = 0.4


    def logistic_activate(self, x):
        return expit(x)

    def softmax_activate(self, inp, i0, n, temp, output):
        sum = 0
        largest = inp.max()
        for i in range(0, n):
            e = np.exp(inp[i + i0] - largest) / temp
            sum += e
            output[i + i0] = e
        for i in range(0, n):
            output[i + i0] /= sum


    def do_nms_sort(self, detections, i0, total, score_thresh, nms_thresh):
        boxes = []
        scores = [None] * total
        for i in range(0, total):
            box_index = i0 + i * (self.classes + self.coords + 1)
            b_width = detections[box_index + 2]
            b_height = detections[box_index + 3]
            b_x = detections[box_index + 0] - b_width / 2 # mb b_width // 2
            b_y = detections[box_index + 1] - b_height / 2
            boxes.append([b_x, b_y, b_width, b_height])

        for k in range(0, self.classes):
            for i in range(0, total):
                box_index = i0 + i * (self.classes + self.coords + 1)
                class_index = box_index + 5
                scores[i] = detections[class_index + k]
                detections[class_index + k] = 0
            indices = cv.dnn.NMSBoxes(boxes, scores, score_thresh, nms_thresh)
            n = len(indices)
            for i in range(0, n):
                # box_index = i0 + indices[i] * (self.classes + self.coords + 1)
                # class_index = box_index + 5
                # detections[class_index + k] = scores[indices[i]]
                box_index = indices[i] * (self.classes + self.coords + 1)
                class_index = box_index + 5
                detections[i0 + class_index + k] = scores[indices[i][0]]

    def forward(self, x):
        cell_size = self.classes + self.coords + 1
        batch_size = x.shape[0]
        rows = x.shape[1]
        cols = x.shape[2]
        sample_size = cell_size * rows * cols * self.anchors
        assert(sample_size * batch_size == x.size)
        hNorm = rows
        wNorm = cols

        src = np.reshape(x, -1)
        dst = np.empty(src.shape)
        for i in range(0, batch_size * rows * cols * self.anchors):
            index = cell_size * i
            data = src[index + 4]
            dst[index + 4] = self.logistic_activate(data)

        for i in range(0, batch_size * rows * cols * self.anchors):
            index = cell_size * i
            self.softmax_activate(src, index + 5, self.classes, 1, dst)

        for b in range(0, batch_size):
            for x in range(0, cols):
                for y in range(0, rows):
                    for a in range(0, self.anchors):
                        index_sample_offset = sample_size * b
                        index = (y * cols + x) * self.anchors + a
                        p_index = index_sample_offset + index * cell_size + 4
                        scale = dst[p_index]
                        # Check classfix !! see opencv scale = 0
                        box_index = index_sample_offset + index * cell_size
                        dst[box_index + 0] = (x + self.logistic_activate(src[box_index + 0])) / cols
                        dst[box_index + 1] = (y + self.logistic_activate(src[box_index + 1])) / rows
                        dst[box_index + 2] = np.exp(src[box_index + 2]) * self.bias[2 * a] / hNorm
                        dst[box_index + 3] = np.exp(src[box_index + 3]) * self.bias[2 * a + 1] / wNorm

                        class_index = index_sample_offset + index * cell_size + 5
                        for j in range(0, self.classes):
                            prob = scale * dst[class_index + j]
                            dst[class_index + j] = prob if (prob > self.thresh) else 0

        if self.nmsthresh > 0:
            for b in range(0, batch_size):
                self.do_nms_sort(dst, b * sample_size, rows * cols * self.anchors, self.thresh, self.nmsthresh)
        return dst.reshape((rows * cols * self.anchors, -1))


if __name__ == "__main__":
    net = Yolo()
    net.load_state_dict(torch.load('yolov2_voc_model'))
    net.eval()
