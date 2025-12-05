#generate dataset
#run our model
#generate eval or mesh

import torch
import numpy as np
import os
from lib.cores import config 
from lib.models.smpl import SMPL_59, get_smpl_faces
from cs5404.visual import render_mesh_gif


with torch.no_grad():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    smpl = SMPL_59(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    updated_smpl_output = np.load(os.path.join(current_dir, 'data', 'random-forest', 'kitro_rf.npy'))
    refined_thetas = torch.from_numpy(updated_smpl_output[:, :-13]).reshape(-1, 24, 3, 3).to(torch.float32)
    refined_shape = torch.from_numpy(updated_smpl_output[:, -13:-3]).to(torch.float32)
    refined_cam = torch.from_numpy(updated_smpl_output[:, -3:]).to(torch.float32)
    updated_smpl = smpl(betas=refined_shape, body_pose=refined_thetas[:, 1:], global_orient=refined_thetas[:, 0].unsqueeze(1), pose2rot=False)
    updated_vertices = updated_smpl.vertices
    
    # Call to generate and save mesh from SMPL model
    # Choice of gif or img save.
    faces = np.array(get_smpl_faces(), dtype = int)
    file_name = "RF_KITRO.gif"
    render_mesh_gif(updated_vertices, faces, file_name)
