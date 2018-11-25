from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from geotnf.transformation import GeometricTnf

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, 
                 train_fe=False, 
                 feature_extraction_cnn='vgg', 
                 normalization=True, 
                 last_layer='', 
                 use_cuda=True):

        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                         'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                         'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                         'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                         'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
            if last_layer=='':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])
        elif feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]
            
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])
        elif feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        elif feature_extraction_cnn == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            self.model = nn.Sequential(*list(self.model.features.children())[:-3])
        elif feature_extraction_cnn == 'densenet161':
            self.model = models.densenet161(pretrained=True)
            self.model = nn.Sequential(*list(self.model.features.children())[:-3])
        elif feature_extraction_cnn == 'densenet169':
            self.model = models.densenet169(pretrained=True)
            self.model = nn.Sequential(*list(self.model.features.children())[:-3])
        elif feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to denseblock3
            self.model = nn.Sequential(*list(self.model.features.children())[:-3])
        if not train_fe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()
 
    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        if self.shape=='3D':
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
            feature_B = feature_B.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B,feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
            # batch x (hB x wB) x hA x wA
        elif self.shape=='4D':
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
            feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A,feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)
       
        correlation_tensor = self.ReLU(correlation_tensor)
        return correlation_tensor
        '''
        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            
        return correlation_tensor
        '''


class FeatureRegression(nn.Module):
    def __init__(self, 
                 output_dim=6, 
                 use_cuda=True, 
                 batch_normalization=True, 
                 kernel_sizes=[7,5], 
                 channels=[128,64],
                 feature_size=15):

        super(FeatureRegression, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i == 0:
                ch_in = feature_size*feature_size
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)        
        self.linear = nn.Linear(ch_out * k_size * k_size, output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    
class CNNGeometric(nn.Module):
    def __init__(self, 
                 output_dim=6, 
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 return_correlation=False,  
                 fr_feature_size=15,
                 fr_kernel_sizes=[7,5],
                 fr_channels=[128,64],
                 feature_self_matching=False,
                 normalize_features=True, 
                 normalize_matches=True, 
                 batch_normalization=True, 
                 train_fe=False,
                 use_cuda=True):
        
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.feature_self_matching = feature_self_matching
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)
        
        self.FeatureCorrelation = FeatureCorrelation(shape='3D',
                                                     normalization=normalize_matches)        
        
        self.FeatureRegression = FeatureRegression(output_dim,
                                                   use_cuda=self.use_cuda,
                                                   feature_size=fr_feature_size,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization)

        self.ReLU = nn.ReLU(inplace=True)
    
    # used only for foward pass at eval and for training with strong supervision
    def forward(self, tnf_batch): 
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        # feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        # regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)
        
        if self.return_correlation:
            return (theta, correlation)
        else:
            return theta

class TwoStageCNNGeometric(CNNGeometric):
    def __init__(self, 
                 fr_feature_size=15,
                 fr_kernel_sizes=[7,5],
                 fr_channels=[128,64],
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 return_correlation=False,                  
                 normalize_features=True,
                 normalize_matches=True, 
                 batch_normalization=True, 
                 bi_directional=True, 
                 train_fe=False,
                 use_cuda=True,
                 s1_output_dim=6,
                 s2_output_dim=18):

        super(TwoStageCNNGeometric, self).__init__(output_dim=s1_output_dim, 
                                                   fr_feature_size=fr_feature_size,
                                                   fr_kernel_sizes=fr_kernel_sizes,
                                                   fr_channels=fr_channels,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   feature_extraction_last_layer=feature_extraction_last_layer,
                                                   return_correlation=return_correlation,
                                                   normalize_features=normalize_features,
                                                   normalize_matches=normalize_matches,
                                                   batch_normalization=batch_normalization,
                                                   train_fe=train_fe,
                                                   use_cuda=use_cuda)
        if s1_output_dim == 6:
            self.geoTnf = GeometricTnf(geometric_model='affine',
                                       use_cuda=use_cuda)
        else:
            tps_grid_size = np.sqrt(s2_output_dim/2)
            self.geoTnf = GeometricTnf(geometric_model='tps', 
                                       tps_grid_size=tps_grid_size, 
                                       use_cuda=use_cuda)
 
        self.FeatureRegression2 = FeatureRegression(output_dim=s2_output_dim,
                                                    use_cuda=use_cuda,
                                                    feature_size=fr_feature_size,
                                                    kernel_sizes=fr_kernel_sizes,
                                                    channels=fr_channels,
                                                    batch_normalization=batch_normalization)        
 
    def mask(self, corr, norm_A, norm_B):
        return torch.max(torch.div(torch.div(corr, norm_A), norm_B), dim=1)[0]

    def forward(self, batch, use_theta_GT_aff=False): 
        #===  STAGE 1 ===#
        f_A = self.FeatureExtraction(batch['source_image'])
        f_B = self.FeatureExtraction(batch['target_image'])

        b,c,h,w = f_A.size()

        correlation_AB = self.FeatureCorrelation(f_A, f_B)
        correlation_AA = self.FeatureCorrelation(f_A, f_A)
        correlation_BA = self.FeatureCorrelation(f_B, f_A)
        correlation_BB = self.FeatureCorrelation(f_B, f_B)
 
        norm_A = torch.norm(f_A, p=2, dim=1).unsqueeze(1).expand(b,h*w,h,w)
        norm_B = torch.norm(f_B, p=2, dim=1).view(b,h*w).unsqueeze(2).unsqueeze(3).expand(b,h*w,h,w)

        mask_AB = self.mask(correlation_AB, norm_A, norm_B)
        mask_AA = self.mask(correlation_AA, norm_A, norm_A)
        mask_BA = self.mask(correlation_BA, norm_B, norm_A)
        mask_BB = self.mask(correlation_BB, norm_B, norm_B)

        correlation_AB = featureL2Norm(correlation_AB)
        correlation_AA = featureL2Norm(correlation_AA)
        correlation_BA = featureL2Norm(correlation_BA)
        correlation_BB = featureL2Norm(correlation_BB)

        #vol_mask_AB = mask_AB.unsqueeze(1).expand_as(correlation_AB)
        #vol_mask_AA = mask_AA.unsqueeze(1).expand_as(correlation_AA)
        #vol_mask_BA = mask_BA.unsqueeze(1).expand_as(correlation_BA)
        #vol_mask_BB = mask_BB.unsqueeze(1).expand_as(correlation_BB)

        #aff_theta_AB = self.FeatureRegression(torch.mul(correlation_AB, vol_mask_AB))
        #aff_theta_AA = self.FeatureRegression(torch.mul(correlation_AA, vol_mask_AA))
        #aff_theta_BA = self.FeatureRegression(torch.mul(correlation_BA, vol_mask_BA))
        #aff_theta_BB = self.FeatureRegression(torch.mul(correlation_BB, vol_mask_BB))
 
        aff_theta_AB = self.FeatureRegression(correlation_AB)
        aff_theta_AA = self.FeatureRegression(correlation_AA)
        aff_theta_BA = self.FeatureRegression(correlation_BA)
        aff_theta_BB = self.FeatureRegression(correlation_BB)
 

        #===  STAGE 2 ===#        
        image_A_wrp = self.geoTnf(batch['source_image'], aff_theta_AB)
        image_B_wrp = self.geoTnf(batch['target_image'], aff_theta_BA)

        f_A_wrp = self.FeatureExtraction(image_A_wrp)
        f_B_wrp = self.FeatureExtraction(image_B_wrp)

        correlation_AwrpB = self.FeatureCorrelation(f_A_wrp, f_B)
        correlation_BAwrp = self.FeatureCorrelation(f_B, f_A_wrp)
        correlation_ABwrp = self.FeatureCorrelation(f_A, f_B_wrp)
        correlation_BwrpA = self.FeatureCorrelation(f_B_wrp, f_A)

        norm_A_wrp = torch.norm(f_A_wrp, p=2, dim=1).unsqueeze(1).expand(b,h*w,h,w)
        norm_B_wrp = torch.norm(f_B_wrp, p=2, dim=1).view(b,h*w).unsqueeze(2).unsqueeze(3).expand(b,h*w,h,w)

        mask_AwrpB = self.mask(correlation_AwrpB, norm_A_wrp, norm_B)
        mask_BAwrp = self.mask(correlation_BAwrp, norm_B, norm_A_wrp)
        mask_ABwrp = self.mask(correlation_ABwrp, norm_A, norm_B_wrp)
        mask_BwrpA = self.mask(correlation_BwrpA, norm_B_wrp, norm_A)

        correlation_AwrpB = featureL2Norm(correlation_AwrpB)
        correlation_BAwrp = featureL2Norm(correlation_BAwrp)
        correlation_ABwrp = featureL2Norm(correlation_ABwrp)
        correlation_BwrpA = featureL2Norm(correlation_BwrpA)

        #vol_mask_AwrpB = mask_AwrpB.unsqueeze(1).expand_as(correlation_AwrpB)
        #vol_mask_BAwrp = mask_BAwrp.unsqueeze(1).expand_as(correlation_BAwrp)
        #vol_mask_ABwrp = mask_ABwrp.unsqueeze(1).expand_as(correlation_ABwrp)
        #vol_mask_BwrpA = mask_BwrpA.unsqueeze(1).expand_as(correlation_BwrpA)

        #tps_theta_AwrpB = self.FeatureRegression2(torch.mul(correlation_AwrpB, vol_mask_AwrpB))
        #tps_theta_BAwrp = self.FeatureRegression2(torch.mul(correlation_BAwrp, vol_mask_BAwrp))
        #tps_theta_ABwrp = self.FeatureRegression2(torch.mul(correlation_ABwrp, vol_mask_ABwrp))
        #tps_theta_BwrpA = self.FeatureRegression2(torch.mul(correlation_BwrpA, vol_mask_BwrpA))

        tps_theta_AwrpB = self.FeatureRegression2(correlation_AwrpB)
        tps_theta_BAwrp = self.FeatureRegression2(correlation_BAwrp)
        tps_theta_ABwrp = self.FeatureRegression2(correlation_ABwrp)
        tps_theta_BwrpA = self.FeatureRegression2(correlation_BwrpA)

        aff_theta = (aff_theta_AB, aff_theta_BA, aff_theta_AA, aff_theta_BB)
        tps_theta = (tps_theta_AwrpB, tps_theta_BAwrp, tps_theta_ABwrp, tps_theta_BwrpA)
        correlation = (correlation_AB, correlation_BA, correlation_AA, correlation_BB, 
                       correlation_AwrpB, correlation_BAwrp, correlation_ABwrp, correlation_BwrpA)
        aff_mask = (mask_AB, mask_BA, mask_AA, mask_BB)
        tps_mask = (mask_AwrpB, mask_BAwrp, mask_ABwrp, mask_BwrpA)

        if self.return_correlation:
            return (aff_theta, tps_theta, aff_mask, tps_mask, correlation)
        else:
            return (aff_theta, tps_theta, aff_mask, tps_mask)
