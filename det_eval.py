import os
from os.path import join, exists, basename, splitext, dirname
import re
import sys
import shapely
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import glob
import json
from tqdm import tqdm
import pickle

def polygon_from_points(points):
    '''

    :param points:array shape is [4,2]
    :return:
    '''


    polygon = Polygon(points).convex_hull
    return polygon


def polygon_iou(poly1, poly2):
    """
    Intersection over union between two shapely polygons.
    """
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            # union_area = poly1.union(poly2).area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def average_precision(rec, prec):
    # source: https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py#L47-L61
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def det_eval(preds, preds_score,labeles, IOU_THRESH=0.5):
    '''
    Evaluation detection by calculating the maximum f-measure across all thresholds
    :param preds: [batch,n,4,2]
    :param preds_score: [batch,n]
    :param labeles: [batch,n1,4,2]
    :param IOU_THRESH:
    :return:
    '''

    assert len(preds)==len(labeles),'predict image is not equal with labels'
    n_gt = labeles.shape[0]*labeles.shape[1]

    # scores and match status of all dts in a single list
    all_dt_match = []



    for i in tqdm(range(len(labeles))):
        # find corresponding gt file
        label=labeles[i]
        pred=preds[i]
        dt_polygons = [polygon_from_points(o) for o in label]
        gt_polygons=[polygon_from_points(o) for o in pred]
        dt_match = []

        for dt_poly in dt_polygons:
            match = False
            for gt_poly in gt_polygons:
                if (polygon_iou(dt_poly, gt_poly) >= IOU_THRESH):
                    match = True
                    break
            dt_match.append(match)
        all_dt_match.extend(dt_match)

        # calculate scores and append to list


    # calculate precision, recall and f-measure at all thresholds
    all_dt_match = np.array(all_dt_match, dtype=np.bool).astype(np.int)
    all_dt_scores = np.array(preds_score)
    sort_idx = np.argsort(all_dt_scores)[::-1]  # sort in descending order
    all_dt_match = all_dt_match[sort_idx]
    all_dt_scores = all_dt_scores[sort_idx]

    n_pos = np.cumsum(all_dt_match)
    n_dt = np.arange(1, len(all_dt_match) + 1)
    precision = n_pos.astype(np.float) / n_dt.astype(np.float)
    recall = n_pos.astype(np.float) / float(n_gt)
    eps = 1e-9
    fmeasure = 2.0 / ((1.0 / (precision + eps)) + (1.0 / (recall + eps)))

    # find maximum fmeasure
    max_idx = np.argmax(fmeasure)

    eval_results = {
        'fmeasure': fmeasure[max_idx],
        'precision': precision[max_idx],
        'recall': recall[max_idx],
        'threshold': all_dt_scores[max_idx],
        'all_precisions': precision,
        'all_recalls': recall
    }

    # evaluation summary
    print('=================================================================')
    print('Maximum f-measure: %f' % eval_results['fmeasure'])
    print('  |-- precision:   %f' % eval_results['precision'])
    print('  |-- recall:      %f' % eval_results['recall'])
    print('  |-- threshold:   %f' % eval_results['threshold'])
    print('=================================================================')


if __name__=='__main__':
    pred=np.array([[[[0,0],[100,0],[100,100],[0,100]]]])
    print(pred.shape)
    pred_score=np.array([[0.8]])
    label=pred
    det_eval(pred,pred_score,pred,IOU_THRESH=0.8)
