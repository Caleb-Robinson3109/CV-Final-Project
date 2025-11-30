"""
dataset.py
This file will get the ground truths and the information created by the KITRO model and aggreated it into on dataset.
It will also have a function that splits the data set randomlly into 5 sections so that we do not test on our training data.
We will be ablt to test 5 different times because we are splitting the data into 5.


the data set is 368 features, + more post kitro refinement
 
pred_theta_j1_11, pred_theta_j1_12, pred_theta_j1_12, 
pred_theta_j1_21, pred_theta_j1_22, pred_theta_j1_23,
pred_theta_j1_31, pred_theta_j1_32, pred_theta_j1_33, ...pred_theta_j24_33,
pred_beta_1, ...pred_beta_10,
pred_camera_1, ...pred_camera10,
intrinsics_11, intrinsics_12, intrinsics13,
intrinsics_21, intrinsics_22, intrinsics23,
intrinsics_31, intrinsics_32, intrinsics33,
keypoints_2d_1_1, keypoints_2d_1_2, ...keypoints_2d_24_1, keypoints_2d_24_2

refined_theta_j1_11, refined_theta_j1_12, refined_theta_j1_13, 
refined_theta_j1_21, refined_theta_j1_22, refined_theta_j1_123,
refined_theta_j1_31, refined_theta_j1_32, refined_theta_j1_33, ...rrefined_theta_j24_33,
refined_shape_1, ... refined_shape_10,
refined_cam_1, ...refined_cam_3,
refined_vertices_1_1, refined_vertices_1_2, refined_vertices_1_3, ...refined_vertices_6890_3

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

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from lib.cores import config
from lib.models.smpl import SMPL_59
from lib.models.kitro import KITRO_refine as kitro_refine_gpu
from lib.models.kitrocpu import KITRO_refine as kitro_refine_cpu

class SMPL_Estimates_Dataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, map_location="cpu")
        self.num_samples = self.data['pred_theta'].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'imgname': self.data['imagename'][idx],
            'pred_theta': self.data['pred_theta'][idx],
            'pred_beta': self.data['pred_beta'][idx],
            'pred_cam': self.data['pred_cam'][idx],
            'intrinsics': self.data['intrinsics'][idx],
            'keypoints_2d': self.data['keypoints_2d'][idx],
            'GT_pose': self.data['GT_pose'][idx],
            'GT_beta': self.data['GT_beta'][idx],
        }


def create_kitro_refinement_dataset():

    path = '../data/refined_both_dataset.pt'

    if os.path.exists(path):
        print(f"{path} already exists. Skipping.")
        return

    data_path = '../data/ProcessedData_CLIFFpred_w2DKP_Both.pt'
    output_path = "../data/refined_both_dataset.pt"
    batch_size_ = 256
    n_refine_loop = 10
    shape_opti_n_iter = 10

    # KITRO options
    kitro_config = {
        'shape_opti_n_iter': shape_opti_n_iter,
        'n_refine_loop': n_refine_loop,
    }

    # Load dataset
    dataset = SMPL_Estimates_Dataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size_, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load SMPL
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    smpl = SMPL_59(config.SMPL_MODEL_DIR, create_transl=False).to(device)

    # Storage for NEW enriched dataset
    output = {
        'imgname': [],
        'pred_theta': [],
        'pred_beta': [],
        'pred_cam': [],
        'intrinsics': [],
        'keypoints_2d': [],
        'GT_pose': [],
        'GT_beta': [],

        # NEW KITRO results
        'refined_thetas': [],
        'refined_shape': [],
        'refined_cam': [],
        'refined_vertices': [],
    }

    # -----------------------
    # Process dataset
    # -----------------------
    pbar = tqdm(dataloader, desc="Running KITRO Refinement")

    for batch in pbar:
        batch_size = batch['pred_theta'].shape[0]

        # Move inputs to device
        init_smpl_estimate = {
            'pred_theta': batch['pred_theta'].to(device),
            'pred_beta': batch['pred_beta'].to(device),
            'pred_cam': batch['pred_cam'].to(device),
        }
        evidence_2d = {
            'intrinsics': batch['intrinsics'].to(device),
            'keypoints_2d': batch['keypoints_2d'].to(device),
        }

        with torch.no_grad():
            refined = kitro_refine_cpu(
                batch_size, 
                init_smpl_estimate=init_smpl_estimate, 
                evidence_2d=evidence_2d, 
                J_regressor=J_regressor, 
                kitro_cfg=kitro_config, 
                smpl=smpl
            ) if device == 'cpu' else kitro_refine_gpu(
                batch_size,
                init_smpl_estimate=init_smpl_estimate,
                evidence_2d=evidence_2d,
                J_regressor=J_regressor,
                kitro_cfg=kitro_config,
                smpl=smpl
            )

            refined_thetas = refined['refined_thetas']
            refined_shape = refined['refined_shape']
            refined_cam = refined['refined_cam']

            # SMPL forward to get refined vertices
            refined_smpl = smpl(
                betas=refined_shape,
                body_pose=refined_thetas[:, 1:],
                global_orient=refined_thetas[:, 0].unsqueeze(1),
                pose2rot=False
            )
            refined_vertices = refined_smpl.vertices

        # Save results (on CPU)
        output['imgname'] += batch['imgname']
        output['pred_theta'].append(batch['pred_theta'])
        output['pred_beta'].append(batch['pred_beta'])
        output['pred_cam'].append(batch['pred_cam'])
        output['intrinsics'].append(batch['intrinsics'])
        output['keypoints_2d'].append(batch['keypoints_2d'])
        output['GT_pose'].append(batch['GT_pose'])
        output['GT_beta'].append(batch['GT_beta'])

        output['refined_thetas'].append(refined_thetas.cpu())
        output['refined_shape'].append(refined_shape.cpu())
        output['refined_cam'].append(refined_cam.cpu())
        output['refined_vertices'].append(refined_vertices.cpu())

    # Stack all tensors
    for k in output.keys():
        if isinstance(output[k], list) and torch.is_tensor(output[k][0]):
            output[k] = torch.cat(output[k], dim=0)

    torch.save(output, output_path)
    print(f"saved enriched dataset to {output_path}")


def save_single_sample():
    input_path = '../data/ProcessedData_CLIFFpred_w2DKP_HM36.pt'
    out_path = '../data/ProcessedData_CLIFFpred_w2DKP_single.pt'

    if os.path.exists(out_path):
        print(f"{out_path} already exists. Skipping.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(input_path, map_location=device)

    single_sample = {}

    for key, value in data.items():
        if torch.is_tensor(value):
            single_sample[key] = value[0:1]  # take first sample, keep batch dim
        elif isinstance(value, list):
            single_sample[key] = [value[0]]  # first element
        else:
            try:
                single_sample[key] = np.expand_dims(value[0], axis=0)  # first sample
            except:
                raise TypeError(f"Unsupported type for key {key}: {type(value)}")

    torch.save(single_sample, out_path)
    print(f"Saved single sample to {out_path}")


def save_small_sample():
    """
    Create a small .pt dataset with the first n_samples from the original.
    Ensures batch dimensions are preserved so KITRO works safely.
    """
    input_path = '../data/ProcessedData_CLIFFpred_w2DKP_HM36.pt'
    out_path = '../data/small.pt'
    n_samples = 1024
    if os.path.exists(out_path):
        print(f"{out_path} already exists. Skipping.")
        return

    # Load original dataset on CPU
    data = torch.load(input_path, map_location='cpu')

    small_sample = {}

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            # Take first n_samples, keep batch dim
            small_sample[key] = value[:n_samples]
        elif isinstance(value, np.ndarray):
            # Convert to torch tensor and keep batch dim
            small_sample[key] = torch.from_numpy(value[:n_samples])
        elif isinstance(value, list):
            # Take first n_samples elements
            small_sample[key] = value[:n_samples]
        else:
            raise TypeError(f"Unsupported type for key {key}: {type(value)}")

    # Save new small dataset
    torch.save(small_sample, out_path)
    print(f"Saved small dataset with {n_samples} samples to {out_path}")

def make_combined_pt():
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


def get_dataset_pre_kitro():
    make_combined_pt()

    data_path = "../data/ProcessedData_CLIFFpred_w2DKP_Both.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(data_path, map_location=torch.device(device))
    
    N = data['pred_theta'].shape[0]
    dataset = []

    for i in range(len(data['imagename'])):
        sample = np.concatenate([
            data['pred_theta'][i].reshape(-1),
            data['pred_beta'][i].reshape(-1),
            data['pred_cam'][i].reshape(-1),
            data['intrinsics'][i].reshape(-1),
            data['keypoints_2d'][i].reshape(-1),
            data['GT_pose'][i].reshape(-1),
            data['GT_beta'][i].reshape(-1),
        ])
    
        dataset.append(sample)

    return np.array(dataset)

def get_dataset_post_kitro():
    make_combined_pt()
    create_kitro_refinement_dataset()

    data_path = "../data/ProcessedData_CLIFFpred_w2DKP_Both.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(data_path, map_location=torch.device(device))
    
    N = data['pred_theta'].shape[0]
    dataset = []

    for i in range(len(data['imagename'])):
        sample = np.concatenate([
            data['pred_theta'][i].reshape(-1),
            data['pred_beta'][i].reshape(-1),
            data['pred_cam'][i].reshape(-1),
            data['intrinsics'][i].reshape(-1),
            data['keypoints_2d'][i].reshape(-1),
            data['refined_thetas'][i].reshape(-1),
            data['refined_shape'][i].reshape(-1),
            data['refined_cam'][i].reshape(-1),
            data['refined_vertices'][i].reshape(-1),
            data['GT_pose'][i].reshape(-1),
            data['GT_beta'][i].reshape(-1),
        ])
    
        dataset.append(sample)

    return np.array(dataset)


def split_dataset(dataset, num_splits: int):
    seed = int(time.time())
    np.random.seed(seed)
    indexes = np.random.permutation(len(dataset))
    splits = np.array_split(indexes, num_splits)
    parts = [dataset[split] for split in splits]
    return parts
    
"""
def main():
    print("main")
    data = get_dataset_pre_kitro()
    print(f"{np.shape(data)}")

if __name__ == "__main__":
    main()
"""