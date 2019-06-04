#definition of IOU based on https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
#definition of AED is our own

import torch
import numpy as np 
import cv2

def iou(prediction, ground_truth):
    ''' Gives sum of IOU score for each image in batch
    '''    
    # We want to score class 0 only, since we want IOU of embryo class
    p = prediction.data.cpu().numpy()[:,0]
    g = ground_truth.data.cpu().numpy()[:,0]
    p = np.array(1*(p>.5), dtype=np.int32)
    g = np.array(1*(g>.5), dtype=np.int32)
    intersection = p&g
    union = p|g
    intersection = intersection.sum((1,2))
    union = union.sum((1,2))
    return intersection/union

def aed(prediction, ground_truth):
    ''' Gives sum of AED score for each image in batch
    '''   
    preds = 1*(prediction.data.cpu().numpy()[:,0]>.5)
    grounds = 1*(ground_truth.data.cpu().numpy()[:,0]>.5)
    score = []
    # For loop through every image in batch
    for j in range(prediction.size()[0]):
        bw_p = np.array(255*(preds[j]), np.uint8)
        bw_g = np.array(255*(grounds[j]), np.uint8)
        nlabels, _, _, _ = cv2.connectedComponentsWithStats(bw_p)
        counts_p = nlabels - 1
        nlabels, _, _, _ = cv2.connectedComponentsWithStats(bw_g)
        counts_g = nlabels - 1
        score.append(abs(counts_p - counts_g)/counts_g)
    return np.array(score,np.int32)