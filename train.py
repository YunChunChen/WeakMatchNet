from __future__ import print_function, division
import os
from os.path import exists, join
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util.dataloader import DataLoader # modified dataloader
from model.cnn_geometric_model import TwoStageCNNGeometric
from model.loss import WeakInlierCount
from model.loss import TwoStageWeakInlierCount
from model.loss import ConsistencyLoss
from model.loss import TransitivityLoss
from data.pf_dataset import PFPascalDataset as PFEval
from data.pf_pascal_dataset import PFPascalDataset
from image.normalization import NormalizeImageDict
from util.torch_util import save_checkpoint
from util.torch_util import BatchTensorToVars
from collections import OrderedDict
import numpy as np
import numpy.random
from scipy.ndimage.morphology import generate_binary_structure
from util.eval_util import compute_metric
from options.options import ArgumentParser
from geotnf.transformation import GeometricTnf
import config

print('WeakAlign training script using weak supervision')

# Argument parsing
args, arg_groups = ArgumentParser(mode='train_weak').parse()
print(args)

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

# load pre-trained model
if args.model != '':
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        
    for name, param in model.FeatureExtraction.state_dict().items():
        model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
    for name, param in model.FeatureRegression.state_dict().items():
        model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])
    for name, param in model.FeatureRegression2.state_dict().items():
        model.FeatureRegression2.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression2.' + name])
        
# set which parts of model to train 
for name,param in model.FeatureExtraction.named_parameters():
    param.requires_grad = False
    if args.train_fe and np.sum([name.find(x)!=-1 for x in args.fe_finetune_params]):
        param.requires_grad = True        
    if args.train_fe and name.find('bn')!=-1 and np.sum([name.find(x)!=-1 for x in args.fe_finetune_params]):
         param.requires_grad = args.train_bn 
            
for name,param in model.FeatureRegression.named_parameters():    
    param.requires_grad = args.train_fr 
    if args.train_fr and name.find('bn')!=-1:
        param.requires_grad = args.train_bn            

for name,param in model.FeatureRegression2.named_parameters():    
    param.requires_grad = args.train_fr 
    if args.train_fr and name.find('bn')!=-1:
        param.requires_grad = args.train_bn

# define loss
print('Using weak loss...')
if args.dilation_filter == 0:
    dilation_filter = 0
else:
    dilation_filter = generate_binary_structure(2, args.dilation_filter)

if args.inlier_loss:
    inliersAffine = WeakInlierCount(geometric_model='affine', seg_mask=args.mask, **arg_groups['weak_loss'])
    inliersComposed = TwoStageWeakInlierCount(use_cuda=use_cuda, seg_mask=args.mask, **arg_groups['weak_loss'])

if args.consistency_loss:
    aff_ConsisLoss = ConsistencyLoss(use_cuda=use_cuda, transform='affine')
    tps_ConsisLoss = ConsistencyLoss(use_cuda=use_cuda, transform='tps')

if args.transitivity_loss:
    aff_TransLoss = TransitivityLoss(use_cuda=use_cuda, transform='affine')
    tps_TransLoss = TransitivityLoss(use_cuda=use_cuda, transform='tps')

if args.consistency_loss or args.transitivity_loss:
    coord = []
    for i in range(config.NUM_OF_COORD):
        for j in range(config.NUM_OF_COORD):
            xx = []
            xx.append(float(i))
            xx.append(float(j))
            coord.append(xx)
    coord = np.expand_dims(np.array(coord).transpose(), axis=0)
    coord = torch.from_numpy(coord).float()

def loss_fun(batch):
    
    aff_theta, tps_theta, aff_mask, tps_mask, correlation = model(batch)

    aff_theta_AB, aff_theta_BA, aff_theta_AA, aff_theta_BB = aff_theta
    tps_theta_AwrpB, tps_theta_BAwrp, tps_theta_ABwrp, tps_theta_BwrpA = tps_theta

    mask_AB, mask_BA, mask_AA, mask_BB = aff_mask
    mask_AwrpB, mask_BAwrp, mask_ABwrp, mask_BwrpA = tps_mask

    corr_AB, corr_BA, corr_AA, corr_BB, corr_AwrpB, corr_BAwrp, corr_ABwrp, corr_BwrpA = correlation

    loss = 0

    if args.consistency_loss:
        aff_AB = aff_ConsisLoss(coord, aff_theta_AB, aff_theta_BA)
        aff_BA = aff_ConsisLoss(coord, aff_theta_BA, aff_theta_AB)
        aff_AA = aff_ConsisLoss(coord, aff_theta_AA, aff_theta_AA)
        aff_BB = aff_ConsisLoss(coord, aff_theta_BB, aff_theta_BB)
        aff_loss = aff_AB + aff_BA + aff_AA + aff_BB
        loss += args.w_consis*aff_loss

    if args.transitivity_loss:
        aff_AA = aff_TransLoss(coord, aff_theta_AB, aff_theta_BA, aff_theta_AA)
        aff_BB = aff_TransLoss(coord, aff_theta_BA, aff_theta_AB, aff_theta_BB)
        aff_loss = aff_AA + aff_BB
        loss += args.w_trans*aff_loss

    if args.inlier_loss:
        inlier_AB = inliersAffine(aff_theta_AB, corr_AB, mask_AB)
        inlier_BA = inliersAffine(aff_theta_BA, corr_BA, mask_BA)

        inlier_AwrpB = inliersComposed(aff_theta_AB, tps_theta_AwrpB, corr_AwrpB, mask_AwrpB)
        inlier_BAwrp = inliersComposed(aff_theta_AB, tps_theta_BAwrp, corr_BAwrp, mask_BAwrp)
        inlier_ABwrp = inliersComposed(aff_theta_BA, tps_theta_ABwrp, corr_ABwrp, mask_ABwrp)
        inlier_BwrpA = inliersComposed(aff_theta_BA, tps_theta_BwrpA, corr_BwrpA, mask_BwrpA)

        aff_score = inlier_AB + inlier_BA
        tps_score = inlier_AwrpB + inlier_BAwrp + inlier_ABwrp + inlier_BwrpA

        inlier_score = aff_score + tps_score
        loss += args.w_inlier*torch.mean(-inlier_score)

    return loss

# dataset 
train_dataset_size = args.train_dataset_size if args.train_dataset_size != 0 else None

dataset = PFPascalDataset(csv_file=os.path.join(config.CSV_PATH, config.TRAIN_DATA),
                          dataset_path=config.DATASET_PATH,
                          transform=NormalizeImageDict(['source_image','target_image']),
                          dataset_size=train_dataset_size,
                          random_crop=args.random_crop)

dataset_eval = PFEval(csv_file=os.path.join(config.CSV_PATH, config.VAL_DATA),
                      dataset_path=config.DATASET_PATH,
                      transform=NormalizeImageDict(['source_image','target_image']))

# filter training categories
if args.categories != 0:
    keep = np.zeros((len(dataset.set),1))
    for i in range(len(dataset.set)):
        keep[i]=np.sum(dataset.set[i]==args.categories)
    keep_idx = np.nonzero(keep)[0]
    dataset.set = dataset.set[keep_idx]
    dataset.img_A_names = dataset.img_A_names[keep_idx]
    dataset.img_B_names = dataset.img_B_names[keep_idx]

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

# dataloader
dataloader = DataLoader(dataset, 
                        batch_size=args.batch_size,
                        shuffle=True, 
                        num_workers=4)
dataloader_eval = DataLoader(dataset_eval, 
                             batch_size=args.batch_size,
                             shuffle=False, 
                             num_workers=4)

checkpoint_name = os.path.join(args.result_model_dir,
                               'inlier_{}_consis_{}_trans_{}.pth.tar'.format(args.w_inlier, args.w_consis, args.w_trans))

# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode == 'train':
            optimizer.zero_grad()
        tnf_batch = batch_preprocessing_fn(batch)
        loss = loss_fn(tnf_batch)
        loss_np = loss.data.cpu().numpy()[0]
        epoch_loss += loss_np
        if mode == 'train':
            loss.backward()
            optimizer.step()
        else:
            loss=None
        if batch_idx % log_interval == 0:
            print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss_np))
    epoch_loss /= len(dataloader)
    print(mode.capitalize()+' set: Average loss: {:.4f}'.format(epoch_loss))
    return epoch_loss

# compute initial value of evaluation metric used for early stopping
if args.eval_metric=='dist':
    metric = 'dist'
if args.eval_metric=='pck':
    metric = 'pck'
do_aff = args.model_aff!=""
do_tps = args.model_tps!=""
two_stage = args.model!='' or (do_aff and do_tps)


if args.categories==0: 
    eval_categories = np.array(range(config.NUM_OF_CLASS))+1
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

# train
best_test_loss = float("inf")

train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

print('Starting training...')

for epoch in range(1, args.num_epochs+1):
    if args.update_bn_buffers==False:
        model.eval()
    else:
        model.train()
    train_loss[epoch-1] = process_epoch('train', epoch, model, loss_fun, optimizer, dataloader, batch_tnf, log_interval=1)
    model.eval()
    stats=compute_metric(metric,model,dataset_eval,dataloader_eval,batch_tnf,args.batch_size,two_stage,do_aff,do_tps,args)
    eval_value=np.mean(stats['aff_tps'][metric][eval_idx])
    print(eval_value)
    
    if args.eval_metric=='pck':
        test_loss[epoch-1] = -eval_value
    else:
        test_loss[epoch-1] = eval_value
        
    # remember best loss
    is_best = test_loss[epoch-1] < best_test_loss
    best_test_loss = min(test_loss[epoch-1], best_test_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer' : optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, is_best,checkpoint_name)

print('Done!')
