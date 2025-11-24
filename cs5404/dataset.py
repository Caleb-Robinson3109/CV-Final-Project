"""
dataset.py
This file will get the ground truths and the information created by the KITRO model and aggreated it into on dataset.
It will also have a function that splits the data set randomlly into 5 sections so that we do not test on our training data.
We will be ablt to test 5 different times because we are splitting the data into 5.


the data set is

image_name, 
pred_theta_j1_11, pred_theta_j1_12, pred_theta_j1_12, 
pred_theta_j1_21, pred_theta_j1_22, pred_theta_j1_23,
pred_theta_j1_31, pred_theta_j1_32, pred_theta_j1_33, ...pred_theta_j24_33,
pred_beta_1, ...pred_beta_10,
pred_camera_1, ...pred_camera10,
intrinsics_11, intrinsics_12, intrinsics13,
intrinsics_21, intrinsics_22, intrinsics23,
intrinsics_31, intrinsics_32, intrinsics33,
keypoints_2d_1_1, keypoints_2d_1_2, ...keypoints_2d_24_1, keypoints_2d_24_2
ground_truth_pose_1, ...ground_truth_pose_72,
ground_truth_beta_1, ...ground_truth_beta_10
"""

import torch
import numpy as np
import time

def get_dataset():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_3dpw = torch.load('../data/ProcessedData_CLIFFpred_w2DKP_3dpw.pt', map_location=torch.device(device))
    dataset_hm36 = torch.load('../data/ProcessedData_CLIFFpred_w2DKP_HM36.pt', map_location=torch.device(device))

    #turn the datasets into 1 np.array

    N1 = dataset_3dpw['pred_theta'].shape[0]
    N2 = dataset_hm36['pred_theta'].shape[0]

    print(f"N1: {N1}, N2: {N2}")

    subset1 = []
    subset2 = [] 

    for i in range(N1):
        sample = np.concatenate([
            dataset_3dpw['pred_theta'][i].reshape(-1),
            dataset_3dpw['pred_beta'][i].reshape(-1),
            dataset_3dpw['pred_cam'][i].reshape(-1),
            dataset_3dpw['intrinsics'][i].reshape(-1),
            dataset_3dpw['keypoints_2d'][i].reshape(-1),
            dataset_3dpw['GT_pose'][i].reshape(-1),
            dataset_3dpw['GT_beta'][i].reshape(-1),
        ])
        subset1.append(sample)

    for i in range(N2):
        sample = np.concatenate([
            dataset_hm36['pred_theta'][i].reshape(-1),
            dataset_hm36['pred_beta'][i].reshape(-1),
            dataset_hm36['pred_cam'][i].reshape(-1),
            dataset_hm36['intrinsics'][i].reshape(-1),
            dataset_hm36['keypoints_2d'][i].reshape(-1),
            dataset_hm36['GT_pose'][i].reshape(-1),
            dataset_hm36['GT_beta'][i].reshape(-1),
        ])
        subset2.append(sample)

    print(f"s1: {len(subset1)} {len(subset1[0])}, s2: {len(subset2)} {len(subset2[0])}")
    dataset = subset1 + subset2
    return np.array(dataset)
    
def split_dataset(dataset, num_splits: int):
    seed = int(time.time())
    np.random.seed(seed)
    indexes = np.random.permutation(len(dataset))
    splits = np.array_split(indexes, num_splits)
    parts = [dataset[split] for split in splits]
    return parts
    

    