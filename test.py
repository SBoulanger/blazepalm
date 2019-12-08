import cv2
import tensorflow as tf
import os, sys
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

from palm_detector import PalmDetector

cam = cv2.VideoCapture(1)
cv2.namedWindow("test")
ret, frame = cam.read()
#input_image_orig = cv2.imread("test.png").astype(np.float32)
input_image = frame.copy()[:,:,::-1]
#input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB).astype(np.float32)
input_image = cv2.resize(input_image,(256,256))
input_image_resized = input_image.copy()
input_image = np.expand_dims(input_image,0)
input_image = np.ascontiguousarray(2 * (input_image/255 -0.5)).astype(np.float32)
print(f"Input Image Shape: {input_image.shape} dtype {input_image.dtype}")
pm = PalmDetector()
pm.build()

print("Loading Model weights")
tf.keras.utils.plot_model(pm.model, 'model.png',show_shapes=True)
for i in range(0,len(pm.layer_names)):
    print(pm.layer_names[i])
    if "residual" in pm.layer_names[i]:
        pm.load_weights_to_residual_block(pm.layer_names[i])
    else:
        layer = pm.model.get_layer(name=pm.layer_names[i])
        if "dconv" in pm.layer_names[i]:
            pm.load_weights_to_layer("weights/"+pm.layer_names[i]+"_w.npy",None,layer)
        else:
            pm.load_weights_to_layer("weights/"+pm.layer_names[i]+"_w.npy","weights/"+pm.layer_names[i]+"_b.npy",layer)
print("done loading weights")

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        #get maximum
        i = order[0]
        #append to keep
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

anchors_path = "anchors.csv"
with open(anchors_path,"r") as csv_f:
    anchors = np.r_[
            [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]

while cam.isOpened():
    ret, frame = cam.read()
    input_image = frame.copy()
    input_image = cv2.resize(input_image,(256,256))
    input_image_resized = input_image.copy()
    input_image = np.expand_dims(input_image,0).astype(np.float32)
    input_image = np.ascontiguousarray(2* (input_image/255 - .5))
    time_s = time.time()
    classificators, regressors = pm.model.predict(input_image)
    classificators = classificators[0]
    regressors = regressors[0]
    #classificators = np.expand_dims(classificators[0,:,0],axis=1)
    print(regressors.shape)
    print(regressors)
    print(classificators.shape)
    print(classificators)
    x1y1 = anchors[:,:2] * 256 - regressors[:,2:4]/2#regressors[:,:2] - regressors[:,2:4]/2
    x2y2 = anchors[:,:2] * 256 + regressors[:,2:4]/2#regressors[:,:2] + regressors[:,2:4]/2
    dets = np.concatenate((x1y1,x2y2),axis=1)
    dets = np.concatenate((dets,classificators),axis=1)

    #dets = np.where(dets[:,4] > 0.5)
    keeps = nms(dets,0.05)
    time_t = (time.time() - time_s) * 1000

    c = 0
    for i in keeps:
        cv2.rectangle(input_image_resized,(int(dets[i,0]),int(dets[i,1])),(int(dets[i,2]),int(dets[i,3])),(0,255,0),2)
        c += 1
        if c > 1:
            break
    print(f"Took {time_t} ms")
    print(f"Anchors {anchors.shape}")
    print(f"Classificators {classificators.shape}")
    print(f"Regressors {regressors.shape}")
    print(f"Detections {len(keeps)}")
    #print(classificators)
    cv2.imshow("test",input_image_resized)
    if cv2.waitKey(1) & 0xff == ord('q'):
        pm.model.save("saved_model.h5")
        break


    continue
    #detection_mask = _sigm(dets[:,-1]) > 0.65
    #candidate_detect  = dets[detection_mask]
    #print(f"Candidates {candidate_detect.shape}")
    #print(candidate_detect)
    #print(candidate_detect)
    candidate_anchors = anchors[detection_mask]

    for i in range(0,len(keeps)): 
        max_idx = np.argmax(candidate_detect[:, 3])

        side = candidate_detect[max_idx,3]
        dx,dy,w,h = candidate_anchors[max_idx, :4]

        center_wo_offset = candidate_anchors[max_idx,:2] * 256

        print(f"dx{dx},dy{dy},center{center_wo_offset},w{w},h{h}")
        center = (int(center_wo_offset[0]),int(center_wo_offset[1]))
        cv2.circle(input_image_resized,center,1,(255,0,0))
        cv2.rectangle(input_image_resized,(int(center_wo_offset[0]-side/2),int(center_wo_offset[1]-side/2)),(int(center_wo_offset[0]+side/2),int(center_wo_offset[1]+side/2)),(0,255,0))
    else:
        print("hand not found")
    cv2.imshow("test",input_image_resized)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
