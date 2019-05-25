import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import cv2 as cv
import torch.optim as optim
from scipy.special import expit
import numpy as np
import math
from net_structure import *


def bbox_iou(boxA, boxB):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    height = max(yi2 - yi1, 0)
    width = max(xi2 - xi1, 0)
    inter_area = height * width

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = max(box1[3] - box1[1], 0) * max(box1[2] - box1[0], 0)
    box2_area = max(box2[3] - box2[1], 0) * max(box2[2] - box2[0], 0)
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area
    return iou



class Yolov2Loss(nn.Module):
    def __init__(self):
        super(Yolov2Loss, self, width = 416, height = 416).__init__()
        self.lcoord = 5
        self.lnoobj = 0.5
        self.grid_s = 13
        self.num_bbox = 5
        self.frameWidth = width
        self.frameHeight = height
        self.thresh = 0.6

    def localization(self, ground_truth, pred):
        local_loss = 0.0
        for cell_ind in range(0, (self.grid_s - 1) ** 2 + 1):
            begin = cell_ind * self.num_bbox
            end = begin + self.num_bbox
            cell_pred = pred[begin: end,:]
            IoU = []
            maxIoU = 0
            ind_max_bb = -1
            ind_max_gt = -1
            for ind_bb, bbox in enumerate(cell_pred, 0):
                for ind_gt, gt in enumerate(ground_truth, 0):
                    curr_IoU = bbox_iou(bbox, gt)
                    if curr_IoU > maxIoU:
                        maxIoU = curr_IoU
                        ind_max_bb = ind_bb
                        ind_max_gt = ind_gt

            if maxIoU != 0:
                max_bbox = cell_pred[ind_max_bb]
                gt = ground_truth[ind_max_gt]
                x_c = max_bbox[0] * self.frameWidth
                y_c = max_bbox[1] * self.frameHeight
                w = max_bbox[2] * self.frameWidth
                h = max_bbox[3] * self.frameHeight
                local_loss += self.lcoord * ( (x_c - gt[0])**2  + (y_c - gt[1])**2 \
                + (math.sqrt(w) - math.sqrt(gt[2]))**2 + (math.sqrt(h) - math.sqrt(gt[3]))**2)

        return local_loss


    def confidence(self, ground_truth, pred):
        conf_loss = 0.0
        for cell_ind in range(0, (self.grid_s - 1) ** 2 + 1):
            begin = cell_ind * self.num_bbox
            end = begin + self.num_bbox
            cell_pred = pred[begin: end,:]
            IoU = []
            maxIoU = 0
            ind_max_bb = -1
            ind_max_gt = -1
            diff_c = []
            for ind_bb, bbox in enumerate(cell_pred, 0):
                c_hat = 0
                ind_c_hat = -1
                for ind_gt, gt in enumerate(ground_truth, 0):
                    curr_IoU = bbox_iou(bbox, gt)
                    if curr_IoU > c_hat:
                        c_hat = curr_IoU
                        ind_c_hat = ind_gt
                    if curr_IoU > maxIoU:
                        maxIoU = curr_IoU
                        ind_max_bb = ind_bb
                        ind_max_gt = ind_gt

                diff_c.append((c_hat * (bbox[4] - 1)) ** 2)
            have_obj = [0] * self.num_bbox
            have_obj[ind_max_bb] = 1
            for i in range(0, self.num_bbox):
                conf_loss += (self.lnoobj * (1 - have_obj[i]) + have_obj[i]) * diff_c[i]

        return conf_loss

    def classification(self, ground_truth, output):
        class_loss = 0.0
        for cell_ind in range(0, (self.grid_s - 1) ** 2 + 1):
            begin = cell_ind * self.num_bbox
            end = begin + self.num_bbox
            cell_pred = output[begin: end,:]
            max_objectness = np.max(cell_pred[:, 4])
            if max_objectness >= self.thresh:
                ind_bbox = np.argmax(cell_pred[:, 4])
                bbox = cell_pred[ind_box, :4]
                maxIoU = 0
                class_gt = -1
                for ind_gt, gt in enumerate(ground_truth, 0):
                    curr_IoU = bbox_iou(bbox, gt)
                    if curr_IoU > maxIoU:
                        maxIoU = curr_IoU
                        class_gt = gt[-1]

                p_hat = [0] * 20
                p_hat[class_gt] = 1
                pred_cl = cell_pred[ind_bbox, 5:]
                pred_cl = pred_cl * cell_pred[ind_bbox, 4]
                for ind, curr_class in enumerate(pred_cl, 0):
                    class_loss += (curr_class - p_hat[ind]) ** 2

        return class_loss

    def forward(self, ground_truth, output):
        pred = output[:, :4]
        loss = localization(ground_truth, pred) + confidence(ground_truth, pred) + classification(ground_truth, output)
        return torch.tensor([loss], requires_grad=True)


transform=transforms.Compose([
     transforms.Resize((416, 416), 2),
     transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (255, 255, 255)) # as 1 / scalefactor in OpenCV
])

root = os.path.join("/", "home", "itlab_sparse_mini", "yolo-9000", "VOCdevkit", "VOC2012", "JPEGImages")
data_train = torchvision.datasets.VOCDetection(root, year='2012', image_set='train', transform=transform)

train_loader = torch.utils.data.DataLoader(data_train,
                                          batch_size=8,
                                          shuffle=True
                                          )

name_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def prepare_ground_truth(target):
    annotation = target['annotation']
    objects = annotation['object']
    ground_truth = []
    for obj in objects:
        class_name = obj['name']
        class_id = name_classes.index(class_name[0])
        bbox = obj['bndbox']
        left = int(bbox['xmin'][0])
        top = int(bbox['ymin'][0])
        right = int(bbox['xmax'][0])
        bottom = int(bbox['ymax'][0])
        w = right - left
        h = bottom - top
        x_c = int(round(left + w / 2))
        y_c = int(round(top + h / 2))
        ground_truth.append([x_c, y_c, w, h, class_id])

    return ground_truth


def train_net():
    net = Yolov2Voc()
    net.train()
    criterion = Yolov2Loss()
    epoch_size = 135
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epoch_size):
        running_loss = 0.0
        for i, (image, target) in enumerate(train_loader, 0):
            ground_truth = prepare_ground_truth(target)
            optimizer.zero_grad()

            output = net(image)
            loss = criterion(ground_truth, output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                       (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    save_folder = os.path.join("/", "home", "itlab_sparse_mini", "my_yolo-pytorch")
    torch.save(net.state_dict(), "train" + '_' +'VOC2012' + '.pth')

if __name__ == '__main__':
    train_net()
