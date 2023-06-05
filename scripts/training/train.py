from NPHM.data.face_dataset import ScannerData
import argparse
import torch
import json, os, yaml
import torch
import numpy as np


from NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from NPHM.models.deepSDF import DeepSDF
from NPHM.models.training import TrainerAutoDecoder
from NPHM import env_paths

parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', type=str)
parser.add_argument('-closed', required=False, action='store_true')
parser.set_defaults(closed=False)
parser.add_argument('-local', required=False, action='store_true')
parser.set_defaults(local=False)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

assert args.cfg_file is not None
CFG = yaml.safe_load(open(args.cfg_file, 'r'))

exp_dir = env_paths.EXPERIMENT_DIR + '/{}/'.format(args.exp_name)
fname = exp_dir + 'configs.yaml'
if not os.path.exists(exp_dir):
    print('Creating checkpoint dir: ' + exp_dir)
    os.makedirs(exp_dir)
    with open(fname, 'w') as yaml_file:
        yaml.safe_dump(CFG, yaml_file, default_flow_style=False)
else:
    with open(fname, 'r') as f:
        print('Loading config file from: ' + fname)
        CFG = yaml.safe_load(f)

print(json.dumps(CFG, sort_keys=True, indent=4))

device = torch.device("cuda")


if args.local:
    lm_inds = np.load(env_paths.ANCHOR_INDICES_PATH)
    anchors = torch.from_numpy(np.load(env_paths.ANCHOR_MEAN_PATH)).float().unsqueeze(0).unsqueeze(0).to(device)
else:
    lm_inds = None
    anchors = None


train_dataset = ScannerData(mode='train',
                            n_supervision_points_face=CFG['training']['npoints_decoder'],
                            n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                            batch_size=CFG['training']['batch_size'],
                            sigma_near=CFG['training']['sigma_near'],
                            lm_inds=lm_inds,
                            is_closed=args.closed)

val_dataset = ScannerData(mode='val',
                          n_supervision_points_face=CFG['training']['npoints_decoder'],
                          n_supervision_points_non_face=CFG['training']['npoints_decoder_non'],
                          batch_size=CFG['training']['batch_size'],
                          sigma_near=CFG['training']['sigma_near'],
                          lm_inds=lm_inds,
                          is_closed=args.closed)


print('Done creating datasets!')

print('Length of Train Dataset: {}'.format(len(train_dataset)))
print('Length of Val Dataset: {}'.format(len(val_dataset)))



if args.local:
    decoder = FastEnsembleDeepSDFMirrored(
        lat_dim_glob=CFG['decoder']['decoder_lat_dim_glob'],
        lat_dim_loc=CFG['decoder']['decoder_lat_dim_loc'],
        hidden_dim=CFG['decoder']['decoder_hidden_dim'],
        n_loc=CFG['decoder']['decoder_nloc'],
        n_symm_pairs=CFG['decoder']['decoder_nsymm_pairs'],
        anchors=anchors,
        n_layers=CFG['decoder']['decoder_nlayers'],
        out_dim=1,
    )
else:
    decoder = DeepSDF(
                lat_dim=CFG['decoder']['decoder_lat_dim'],
                hidden_dim=CFG['decoder']['decoder_hidden_dim'],
                geometric_init=True,
                out_dim=1,
                )

#name for wandb
project = 'shape_space'

decoder = decoder.to(device)

trainer = TrainerAutoDecoder(decoder, CFG, device, train_dataset, val_dataset, args.exp_name, project,
                                      is_closed=args.closed)
if 'nepochs' in CFG['training']:
    trainer.train_model(CFG['training']['nepochs'])
else:
    trainer.train_model(30001)
