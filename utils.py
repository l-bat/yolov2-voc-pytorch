import cv2 as cv
import numpy as np

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def preprocessing(frame, inpWidth = 416, inpHeight = 416):
    blob = cv.dnn.blobFromImage(frame, 1.0/255, (inpWidth, inpHeight), mean=[0, 0, 0], swapRB=False, crop=False)
    return blob

# OpenCV postprocess for Region layer
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    print("frame: ", frame.shape)
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    label = '%.2f' % conf

    # Print a label of class.
    if classes:
     assert(classId < len(classes))
     label = '%s: %s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def write_to_file(filename, classId, confidence, left, top, right, bottom):
    dir = "my_result"
    name = os.path.join(dir, filename)
    file = open(name, "w+")
    line = classes[classId] + " " + str(confidence) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + "\n"
    file.write(line)
    file.close()

def postprocess(frame, out, write=False, filename='img.jpeg'):
    confThreshold = 0.5
    nmsThreshold = 0.4

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > confThreshold:
            center_x = int(round(detection[0] * frameWidth))
            center_y = int(round(detection[1] * frameHeight))
            width = int(round(detection[2] * frameWidth))
            height = int(round(detection[3] * frameHeight))
            left = int(round(center_x - width / 2))
            top = int(round(center_y - height / 2))
            classIds.append(classId)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if write:
            write_to_file(filename, classIds[i], confidences[i], left, top, left + width, top + height)
        else:
            drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
