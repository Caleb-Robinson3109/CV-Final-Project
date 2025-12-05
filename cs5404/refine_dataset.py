import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from lib.cores import config
from lib.models.smpl import SMPL_59
from lib.models.kitrocpu import KITRO_refine


# -----------------------
# Dataset loader
# -----------------------
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


# -----------------------
# Main refinement + saving
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= 'data/ProcessedData_CLIFFpred_w2DKP_Both.pt',help=".pt file with original SMPL estimates")
    #parser.add_argument('--data_path', type=str, default= 'data/small.pt',help=".pt file with original SMPL estimates")
    parser.add_argument('--output_path', type=str, default="data/refined_both_dataset.pt", help="Where to save enriched dataset")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_refine_loop', type=int, default=10)
    parser.add_argument('--shape_opti_n_iter', type=int, default=10)

    args = parser.parse_args()

    # KITRO options
    kitro_config = {
        'shape_opti_n_iter': args.shape_opti_n_iter,
        'n_refine_loop': args.n_refine_loop,
    }

    # Load dataset
    dataset = SMPL_Estimates_Dataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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
            refined = KITRO_refine(
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

    torch.save(output, args.output_path)
    print(f"saved enriched dataset to {args.output_path}")
