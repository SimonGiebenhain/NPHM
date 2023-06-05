from NPHM.models.deepSDF import DeepSDF, DeformationNetwork
from NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from NPHM import env_paths
from NPHM.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from NPHM.models.reconstruction import deform_mesh, get_logits, get_logits_backward
from NPHM.models.fitting import inference_iterative_root_finding_joint, inference_identity_space
from NPHM.data.manager import DataManager

import numpy as np
import argparse
import json, yaml
import os
import os.path as osp
import torch
import pyvista as pv
import trimesh


parser = argparse.ArgumentParser(
    description='Run generation'
)

parser.add_argument('-resolution' , default=256, type=int)
parser.add_argument('-batch_points', default=20000, type=int)
parser.add_argument('-cfg_file', type=str, required=True)
parser.add_argument('-exp_name', type=str, required=True)
parser.add_argument('-exp_tag', type=str, required=True)
parser.add_argument('-demo', required=False, action='store_true')
parser.set_defaults(demo=False)
parser.add_argument('-sample', required=False, action='store_true')
parser.set_defaults(sample=False)


args = parser.parse_args()

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

with open(args.cfg_file, 'r') as f:
    print('Loading config file from: ' + args.cfg_file)
    CFG = yaml.safe_load(f)

print(json.dumps(CFG, sort_keys=True, indent=4))



weight_dir_shape = env_paths.EXPERIMENT_DIR + '/{}/'.format(CFG['exp_name_shape'])
if CFG['exp_name_expr'] is not None:
    weight_dir_expr = env_paths.EXPERIMENT_DIR + '/{}/'.format(CFG['exp_name_expr'])

# load config files
fname_shape = weight_dir_shape + 'configs.yaml'
with open(fname_shape, 'r') as f:
    print('Loading config file from: ' + fname_shape)
    CFG_shape = yaml.safe_load(f)
if CFG['exp_name_expr'] is not None:
    fname_expr = weight_dir_expr + 'configs.yaml'
    with open(fname_expr, 'r') as f:
        print('Loading config file from: ' + fname_expr)
        CFG_expr = yaml.safe_load(f)

device = torch.device("cuda")

print('###########################################################################')
print('####################     Shape Model Configs     #############################')
print('###########################################################################')
print(json.dumps(CFG_shape, sort_keys=True, indent=4))

if CFG['exp_name_expr'] is not None:
    print('###########################################################################')
    print('####################     Expression Model Configs     #############################')
    print('###########################################################################')
    print(json.dumps(CFG_expr, sort_keys=True, indent=4))


if CFG['local_shape']:
    lm_inds = np.load(env_paths.ANCHOR_INDICES_PATH)
    anchors = torch.from_numpy(np.load(env_paths.ANCHOR_MEAN_PATH)).float().unsqueeze(0).unsqueeze(0).to(device)
else:
    lm_inds = None
    anchors = None


if CFG['local_shape']:
    decoder_shape = FastEnsembleDeepSDFMirrored(
        lat_dim_glob=CFG_shape['decoder']['decoder_lat_dim_glob'],
        lat_dim_loc=CFG_shape['decoder']['decoder_lat_dim_loc'],
        hidden_dim=CFG_shape['decoder']['decoder_hidden_dim'],
        n_loc=CFG_shape['decoder']['decoder_nloc'],
        n_symm_pairs=CFG_shape['decoder']['decoder_nsymm_pairs'],
        anchors=anchors,
        n_layers=CFG_shape['decoder']['decoder_nlayers'],
        pos_mlp_dim=CFG_shape['decoder'].get('pos_mlp_dim', 256),
    )
else:
    decoder_shape = DeepSDF(
        lat_dim=CFG_shape['decoder']['decoder_lat_dim'],
        hidden_dim=CFG_shape['decoder']['decoder_hidden_dim'],
        geometric_init=True
    )

decoder_shape = decoder_shape.to(device)


if CFG['exp_name_expr'] is not None:
    if CFG['local_shape']:
        decoder_expr = DeformationNetwork(mode=CFG_expr['ex_decoder']['mode'],
                                 lat_dim_expr=CFG_expr['ex_decoder']['decoder_lat_dim_expr'],
                                 lat_dim_id=CFG_expr['ex_decoder']['decoder_lat_dim_id'],
                                 lat_dim_glob_shape=CFG_expr['id_decoder']['decoder_lat_dim_glob'],
                                 lat_dim_loc_shape=CFG_expr['id_decoder']['decoder_lat_dim_loc'],
                                 n_loc=CFG_expr['id_decoder']['decoder_nloc'],
                                 anchors=anchors,
                                 hidden_dim=CFG_expr['ex_decoder']['decoder_hidden_dim'],
                                 nlayers=CFG_expr['ex_decoder']['decoder_nlayers'],
                                 input_dim=3, out_dim=3
                                 )
    else:
        decoder_expr = DeepSDF(lat_dim=512+200,
                               hidden_dim=1024,
                               out_dim=3)
    decoder_expr = decoder_expr.to(device)


path = osp.join(weight_dir_shape, 'checkpoints/checkpoint_epoch_{}.tar'.format(CFG['checkpoint_shape']))
print('Loaded checkpoint from: {}'.format(path))
checkpoint = torch.load(path, map_location=device)
decoder_shape.load_state_dict(checkpoint['decoder_state_dict'], strict=True)


if 'latent_codes_state_dict' in checkpoint:
    n_train_subjects = checkpoint['latent_codes_state_dict']['weight'].shape[0]
    n_val_subjects = checkpoint['latent_codes_val_state_dict']['weight'].shape[0]
    if CFG['local_shape']:
        latent_codes_shape = torch.nn.Embedding(n_train_subjects, decoder_shape.lat_dim)
        latent_codes_shape_val = torch.nn.Embedding(n_val_subjects, decoder_shape.lat_dim)
    else:
        latent_codes_shape = torch.nn.Embedding(n_train_subjects, 512)
        latent_codes_shape_val = torch.nn.Embedding(n_val_subjects, 512)

    latent_codes_shape.load_state_dict(checkpoint['latent_codes_state_dict'])
    latent_codes_shape_val.load_state_dict(checkpoint['latent_codes_val_state_dict'])
else:
    latent_codes_shape = None
    latent_codes_shape_val = None

if CFG['exp_name_expr'] is not None:
    path = osp.join(weight_dir_expr, 'checkpoints/checkpoint_epoch_{}.tar'.format(CFG['checkpoint_expr']))
    print('Loaded checkpoint from: {}'.format(path))
    checkpoint = torch.load(path, map_location=device)
    decoder_expr.load_state_dict(checkpoint['decoder_state_dict'], strict=True)
    if 'latent_codes_state_dict' in checkpoint:
        latent_codes_expr = torch.nn.Embedding(checkpoint['latent_codes_state_dict']['weight'].shape[0], 200)
        latent_codes_expr.load_state_dict(checkpoint['latent_codes_state_dict'])
        latent_codes_expr_val = torch.nn.Embedding(checkpoint['latent_codes_val_state_dict']['weight'].shape[0], 200)
        latent_codes_expr_val.load_state_dict(checkpoint['latent_codes_val_state_dict'])
    else:
        latent_codes_expr = None
        latent_codes_expr_val = None
else:
    decoder_expr = None


mini = [-.55, -.5, -.95]
maxi = [0.55, 0.75, 0.4]
grid_points = create_grid_points_from_bounds(mini, maxi, args.resolution)
grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float)
grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).to(device)


#TODO
out_dir = env_paths.FITTING_DIR + '/forward_{}/{}/'.format(args.exp_name, args.exp_tag)


os.makedirs(out_dir, exist_ok=True)
fname = out_dir + 'configs.yaml'
with open(fname, 'w') as yaml_file:
    yaml.safe_dump(CFG, yaml_file, default_flow_style=False)



def sample_shape_space():

    if CFG['local_shape']:
        out_dir = 'nphm_shape_space_samples_085'
    else:
        out_dir = 'npm_shape_space_samples_085'
    print(f'Saving Random Samples in {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    step = 0


    if False and latent_codes_shape is not None:
        lat_mean = torch.mean(latent_codes_shape.weight, dim=0)
        lat_std = torch.std(latent_codes_shape.weight, dim=0)
    else:
        if CFG['local_shape']:
            lat_mean = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_mean.npy'))
            lat_std = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_std.npy'))
        else:
            lat_mean = torch.from_numpy(np.load(env_paths.ASSETS + 'npm_lat_mean.npy'))
            lat_std = torch.from_numpy(np.load(env_paths.ASSETS + 'npm_lat_std.npy'))
    for i in range(100):
        lat_rep = (torch.randn(lat_mean.shape) * lat_std * 0.85 + lat_mean).cuda()

        logits = get_logits(decoder_shape, lat_rep, grid_points, nbatch_points=25000)
        print('starting mcubes')

        mesh = mesh_from_logits(logits, mini, maxi, args.resolution)
        print('done mcubes')

        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh)
        pl.reset_camera()
        pl.camera.position = (0, 0, 3)
        pl.camera.zoom(1.4)
        pl.set_viewup((0, 1, 0))
        pl.camera.view_plane_normal = (-0, -0, 1)
        #pl.show()
        pl.show(screenshot=out_dir + '/step_{:04d}.png'.format(step))
        mesh.export(out_dir + '/mesh_{:04d}.ply'.format(step))
        #print(pl.camera)
        step += 1


def fit_iterative_rootfinding():
    decoder_expr.eval()
    np.random.seed(0)
    torch.manual_seed(0)

    manager = DataManager()

    subjects = env_paths.subjects_test
    if args.demo:
        subjects = [351, 365]
        env_paths.DATA_SINGLE_VIEW = env_paths.DUMMY_single_view
        env_paths.DATA = env_paths.DUMMY_DATA
    print('############ Starting Fitting ############')
    for subj in subjects:
        print('Fitting subject {}'.format(subj))

        inds = manager.get_expressions(subj, testing=True)
        print('There are expressions with the following indices: {}'.format(inds))
        all_obs = []
        for k, expr_ind in enumerate(inds):
            print('Fitting subject {}, expression {}'.format(subj, expr_ind))
            point_cloud = manager.get_single_view_obs(subj, expr_ind, include_back=k==0)
            obs = torch.from_numpy(point_cloud).float().to(device)
            all_obs.append(obs)

        lambdas_expr = {'surface': 2.0,
                        'reg_expr': 0.01,
                        'reg_global': 0.25,
                        'reg_unobserved': 10,
                        'reg_loc': 0.05,
                        'symm_dist': 5.0,
                        }

        schedule_cfg = {'lr': {200: 2, 400: 2, 600: 2, 800: 2},
                        'symm_dist': {200: 10, 500: 9999},
                        'reg_glob': {200: 3, 600: 10},
                        'reg_loc': {500: 3, 600: 10},
                        'reg_expr': {600: 10},
                        }

        decoder_shape.train()
        step_scale = 1 / 1  # factor to reduce number of steps
        lat_reps_expr, lat_rep_shape, anchors = inference_iterative_root_finding_joint(decoder_shape,
                                                                                       decoder_expr,
                                                                                       all_obs,
                                                                                       lambdas_expr,
                                                                                       schedule_cfg=schedule_cfg,
                                                                                       n_steps=1000,
                                                                                       step_scale=step_scale)
        decoder_shape.eval()

        logits = get_logits(decoder_shape, lat_rep_shape, grid_points, nbatch_points=args.batch_points)
        mesh_can = mesh_from_logits(logits, mini, maxi, args.resolution)
        for i, expr_ind in enumerate(inds):
            mesh = deform_mesh(mesh_can, decoder_expr, lat_reps_expr[i, ...].unsqueeze(0), anchors, lat_rep_shape=lat_rep_shape)
            mesh.export(out_dir +  '{}_{}.ply'.format(subj, expr_ind))
            np.save(out_dir + '{}_{}_lat_shape.npy'.format(subj, expr_ind),
                    lat_rep_shape.detach().cpu().numpy())
            np.save(out_dir + '{}_{}_lat_expr.npy'.format(subj, expr_ind),
                    lat_reps_expr[i, ...].unsqueeze(0).detach().cpu().numpy())

        torch.cuda.empty_cache()



if __name__ == '__main__':
    if args.sample:
        sample_shape_space()
    else:
        fit_iterative_rootfinding()



