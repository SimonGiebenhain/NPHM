from NPHM.data.face_dataset import ScannerDeformatioData
from NPHM.models import training_corresp as training
import argparse
import torch
import json, os, yaml
import torch
import numpy as np

from NPHM.models.deepSDF import DeformationNetwork, DeepSDF
from NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from NPHM import env_paths


parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-cfg_file', type=str)
parser.add_argument('-ckpt', type=int)
parser.add_argument('-mode', required=True, type=str)



try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

assert args.cfg_file is not None
CFG = yaml.safe_load(open(args.cfg_file, 'r'))

CFG['ex_decoder']['mode'] = args.mode

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

lm_inds = np.load(env_paths.ANCHOR_INDICES_PATH)
anchors = torch.from_numpy(np.load(env_paths.ANCHOR_MEAN_PATH)).float().unsqueeze(0).unsqueeze(0).to(device)


train_dataset = ScannerDeformatioData('train',
                                      CFG['training']['npoints_decoder'],
                                      CFG['training']['batch_size'],
                                      lm_inds=lm_inds
                                      )
val_dataset = ScannerDeformatioData('val',
                                    CFG['training']['npoints_decoder'],
                                    CFG['training']['batch_size'],
                                    lm_inds=lm_inds
                                    )
print('Lens of datasets right after creation')
print(len(train_dataset))
print(len(val_dataset))



if args.mode == 'npm':
    decoder = DeepSDF(lat_dim=512+200,
                      hidden_dim=1024,
                      geometric_init=False,
                      out_dim=3,
                      input_dim=3)
    decoder.lat_dim_expr = 200
else:
    decoder = DeformationNetwork(mode=CFG['ex_decoder']['mode'],
                             lat_dim_expr=CFG['ex_decoder']['decoder_lat_dim_expr'],
                             lat_dim_id=CFG['ex_decoder']['decoder_lat_dim_id'],
                             lat_dim_glob_shape=CFG['id_decoder']['decoder_lat_dim_glob'],
                             lat_dim_loc_shape=CFG['id_decoder']['decoder_lat_dim_loc'],
                             n_loc=39,
                             anchors=anchors,
                             hidden_dim=CFG['ex_decoder']['decoder_hidden_dim'],
                             nlayers=CFG['ex_decoder']['decoder_nlayers'],
                             out_dim=3,
                             input_dim=3,
                             )


if 'shape_exp_name' in CFG['training']:
    if args.mode == 'npm':
        decoder_shape = DeepSDF(
            lat_dim=CFG['id_decoder']['decoder_lat_dim'],
            hidden_dim=CFG['id_decoder']['decoder_hidden_dim'],
            geometric_init=True,
            out_dim=1,
        )
        decoder_shape.mlp_pos = None

    else:
        decoder_shape = FastEnsembleDeepSDFMirrored(
            lat_dim_glob=CFG['id_decoder']['decoder_lat_dim_glob'],
            lat_dim_loc=CFG['id_decoder']['decoder_lat_dim_loc'],
            hidden_dim=CFG['id_decoder']['decoder_hidden_dim'],
            n_loc=CFG['id_decoder']['decoder_nloc'],
            n_symm_pairs=CFG['id_decoder']['decoder_nsymm_pairs'],
            anchors=anchors,
            n_layers=CFG['id_decoder']['decoder_nlayers'],
            out_dim=1,
    )

else:
    decoder_shape = None


project = 'scanner_deformations'

print('Len of train dataset: {}'.format(len(train_dataset)))
print('Len of vl dataset: {}'.format(len(val_dataset)))



trainer = training.TrainerAutoDecoder(decoder,
                                      decoder_shape,
                                      CFG,
                                      device,
                                      train_dataset,
                                      val_dataset,
                                      args.exp_name,
                                      project,
                                      ckpt=args.ckpt)
trainer.train_model(8000)
