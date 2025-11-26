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
from tqdm import tqdm 

import sys
import os

sys.path.append(os.path.abspath(".."))

import eval_KITRO

def make_dataset():
    path_3dpw = '../data/ProcessedData_CLIFFpred_w2DKP_3dpw.pt'
    path_hm36 = '../data/ProcessedData_CLIFFpred_w2DKP_HM36.pt'
    out_path = "../data/ProcessedData_CLIFFpred_w2DKP_Both.pt"

    if os.path.exists(out_path):
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data1 = torch.load(path_3dpw, map_location=torch.device(device))
    data2 = torch.load(path_hm36, map_location=torch.device(device))

    merged = {}

    for key in data1:
        if key not in data2:
            raise ValueError(f"Key {key} missing in second file")

        v1 = data1[key]
        v2 = data2[key]

        if torch.is_tensor(v1):
            merged[key] = torch.cat([v1, v2], dim=0)

        elif isinstance(v1, list):
            merged[key] = v1 + v2

        else:
            try:
                merged[key] = np.concatenate([v1, v2], axis=0)
            except:
                raise TypeError(f"Unsupported type for key {key}: {type(v1)}")

    torch.save(merged, out_path)


def get_dataset():

    #get the dataset/ create a merged data set if needed
    make_dataset

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #dataset = torch.load('../data/ProcessedData_CLIFFpred_w2DKP_Both.pt', map_location=torch.device(device))
    
    kitro_config = {
        'shape_opti_n_iter': 10,
        'n_refine_loop': 10,
    }
    
    data_path = '../data/ProcessedData_CLIFFpred_w2DKP_Both.pt'
    dataset = eval_KITRO.SMPL_Estimates_Dataset(data_path)
    dataloader = eval_KITRO.DataLoader(dataset, batch_size=256, shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #run the refinedment part of kitro and add it to the dataset
    J_regressor = torch.from_numpy(np.load(eval_KITRO.config.JOINT_REGRESSOR_H36M)).float()
    smpl = eval_KITRO.SMPL_59(eval_KITRO.config.SMPL_MODEL_DIR, create_transl=False).to(device)

    pbar = eval_KITRO.tqdm(dataloader, desc='Processing')
    i = 0
    for batch in pbar:
        # batch should contain dictionaries with keys: 'pred_theta', 'pred_beta', 'pred_cam', 'intrinsics', 'keypoints_2d', 'GT_pose', 'GT_beta'
        curr_batch_size = batch['pred_theta'].shape[0]
        init_smpl_estimate = {
            'pred_theta': batch['pred_theta'].to(device),  # Predicted 3D rotation matrix (shape: [samples, 24, 3, 3])
            'pred_beta': batch['pred_beta'].to(device),    # Predicted body shape parameters (shape: [b, 10])
            'pred_cam': batch['pred_cam'].to(device),      # Predicted camera translation (shape: [b, 3])
        }
        evidence_2d = {
            'intrinsics': batch['intrinsics'].to(device),  # Intrinsic camera parameters (shape: [b, 3, 3])
            'keypoints_2d': batch['keypoints_2d'].to(device),  # 2D keypoints (shape: [b, 24, 2])
        }
        with torch.no_grad():
            updated_smpl_output = eval_KITRO.KITRO_refine(curr_batch_size, init_smpl_estimate=init_smpl_estimate, evidence_2d=evidence_2d, J_regressor=J_regressor, kitro_cfg=kitro_config, smpl=smpl)
            refined_thetas = updated_smpl_output['refined_thetas']
            refined_shape = updated_smpl_output['refined_shape']
            refined_cam = updated_smpl_output['refined_cam']
            updated_smpl = smpl(betas=refined_shape, body_pose=refined_thetas[:, 1:], global_orient=refined_thetas[:, 0].unsqueeze(1), pose2rot=False)
            updated_vertices = updated_smpl.vertices

            print(f"{i}: {updated_smpl_output.keys}, {updated_smpl}")
        i = i +1

        


    #turn the datasets into 1 np.array
"""
    N = dataset['pred_theta'].shape[0]

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
    """
    
def split_dataset(dataset, num_splits: int):
    seed = int(time.time())
    np.random.seed(seed)
    indexes = np.random.permutation(len(dataset))
    splits = np.array_split(indexes, num_splits)
    parts = [dataset[split] for split in splits]
    return parts
    

def main():
    print("hello!")
    make_dataset()
    get_dataset()

if __name__ == "__main__":
    main()