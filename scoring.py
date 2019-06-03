#definition of IOU based on https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

import torch
import numpy as np 

def iou(prediction, ground_truth):
    # We want to score class 0 only, since we want IOU of embryo class
    p = prediction.cpu().numpy()[:,0]
    g = ground_truth.cpu().numpy()[:,0]
    p = np.array(1*(p>.5), dtype=np.int32)
    g = np.array(1*(g>.5), dtype=np.int32)
    intersection = p&g
    union = p|g
    intersection = intersection.sum((1,2))
    union = union.sum((1,2))
    return intersection/union