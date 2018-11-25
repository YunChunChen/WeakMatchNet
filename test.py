from __future__ import print_function, division
import os
from os.path import exists, join
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util.dataloader import DataLoader # modified dataloader
from model.cnn_geometric_model import TwoStageCNNGeometric
from data.pf_dataset import PFPascalDataset as PFEval
from image.normalization import NormalizeImageDict
from util.torch_util import save_checkpoint
from util.torch_util import BatchTensorToVars
from collections import OrderedDict
import numpy as np
from util.eval_util import compute_metric
from options.options import ArgumentParser
from geotnf.transformation import GeometricTnf
import config

args, arg_groups = ArgumentParser(mode='train_weak').parse()
torch.cuda.set_device(args.gpu)

use_cuda = torch.cuda.is_available()

# Seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# CNN model and loss
print('Creating CNN model...')

model = TwoStageCNNGeometric(use_cuda=use_cuda,
                             return_correlation=True,
                             bi_directional=args.bi_directional,
                             **arg_groups['model'])

if args.model != '':
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        
    for name, param in model.FeatureExtraction.state_dict().items():
        model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
    for name, param in model.FeatureRegression.state_dict().items():
        model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])
    for name, param in model.FeatureRegression2.state_dict().items():
        model.FeatureRegression2.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression2.' + name])
        
dataset_eval = PFEval(csv_file=os.path.join(args.eval_dataset_path, config.TEST_DATA),
                      dataset_path=args.eval_dataset_path,
                      transform=NormalizeImageDict(['source_image','target_image']))

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

dataloader_eval = DataLoader(dataset_eval, 
                             batch_size=args.batch_size,
                             shuffle=False, 
                             num_workers=4)

if args.eval_metric=='dist':
    metric = 'dist'
if args.eval_metric=='pck':
    metric = 'pck'
do_aff = args.model_aff!=""
do_tps = args.model_tps!=""
two_stage = args.model!='' or (do_aff and do_tps)

if args.categories==0: 
    eval_categories = np.array(range(20))+1
else:
    eval_categories = np.array(args.categories)

eval_flag = np.zeros(len(dataset_eval))
for i in range(len(dataset_eval)):
    eval_flag[i]=sum(eval_categories==dataset_eval.category[i])
eval_idx = np.flatnonzero(eval_flag)

model.eval()

stats=compute_metric(metric,model,dataset_eval,dataloader_eval,batch_tnf,args.batch_size,two_stage,do_aff,do_tps,args)
eval_value=np.mean(stats['aff_tps'][metric][eval_idx])

print(eval_value)
