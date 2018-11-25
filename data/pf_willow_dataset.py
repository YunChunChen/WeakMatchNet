from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from geotnf.point_tnf import PointTnf, PointsToUnitCoords, PointsToPixelCoords
    
class PFWillowDataset(Dataset):
    
    """
        Description:
            Proposal Flow image pair dataset

        Args:
            csv_file (string): Path to the csv file with image names and transformations.
            dataset_path (string): Directory with the images.
            output_size (2-tuple): Desired output size
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
    """

    def __init__(self, csv_file, dataset_path, output_size=(240,240), transform=None, category=None):

        self.category_names = ['car(G)', 'car(M)', 'car(S)', 'duck(S)',
                               'motorbike(G)', 'motorbike(M)', 'motorbike(S)', 
                               'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']
        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.category = self.pairs.iloc[:,2].as_matrix().astype('float')
        if category is not None:
            cat_idx = np.nonzero(self.category==category)[0]
            self.category=self.category[cat_idx]
            self.pairs=self.pairs.iloc[cat_idx,:]
        self.img_A_names = self.pairs.iloc[:,0]
        self.img_B_names = self.pairs.iloc[:,1]
        self.point_A_coords = self.pairs.iloc[:, 3:5]
        self.point_B_coords = self.pairs.iloc[:, 5:7]
        self.flip = self.pairs.iloc[:,7].as_matrix().astype('int')
        self.dataset_path = dataset_path         
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False) 
        """ Newly added """
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))
        self.pointTnf = PointTnf(use_cuda=False) 
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        flipA = False
        flipB = False
        if self.flip[idx] == 1:
            flipB = True
        elif self.flip[idx] == 2:
            flipA = True
        elif self.flip[idx] == 3:
            flipA = True
            flipB = True
        image_A, im_size_A = self.get_image(self.img_A_names, idx, flip=flipA)
        image_B, im_size_B = self.get_image(self.img_B_names, idx, flip=flipB)

        # category: class of pf-pascal, will be the index of class list plus 1
        image_category = self.category[idx]

        # get pre-processed point coords
        point_A_coords, warped_point_A_coords = self.get_points(self.point_A_coords, idx, flipA, im_size_A, (240,240,3))
        #print("point_A_coords size:", point_A_coords.size)
        #print("warped point_A_coords size:", warped_point_A_coords.size)
        point_B_coords, warped_point_B_coords = self.get_points(self.point_B_coords, idx, flipB, im_size_B, (240,240,3))
        
        correspondence = self.pack_corr(point_A_coords, warped_point_A_coords, warped_point_B_coords)

        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        L_pck = torch.FloatTensor([torch.max(point_A_coords.max(1)[0] - point_A_coords.min(1)[0])])
         
        sample = {'source_image': image_A, 
                  'target_image': image_B, 
                  'source_im_size': im_size_A, 
                  'target_im_size': im_size_B, 
                  'source_points': point_A_coords, 
                  'target_points': point_B_coords, 
                  'warped_source_points': warped_point_A_coords, 
                  'warped_target_points': warped_point_B_coords, 
                  'correspondence': correspondence,
                  'category': image_category,
                  'L_pck': L_pck}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx, flip):
        img_name = os.path.join(self.dataset_path, img_name_list.iloc[idx])
        image = io.imread(img_name)
        
        # if gray scale, convert to 3-channel image
        if image.ndim == 2:
            image = np.repeat(np.expand_dim(image, 2), axis=2, repeats=3)
        
        if flip:
            image = np.flip(image, 1)

        # get image size
        im_size = np.asarray(image.shape)
        
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        
        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)
        
        im_size = torch.Tensor(im_size.astype(np.float32))
        
        return (image, im_size)
    
    def get_points(self, point_coords_list, idx, flip, im_size, warped_im_size):
        X = np.fromstring(point_coords_list.iloc[idx,0], sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx,1], sep=';')
        if flip:
            X = im_size[1] - X
        Xpad = -np.ones(20); Xpad[:len(X)] = X
        Ypad = -np.ones(20); Ypad[:len(X)] = Y
        point_coords = np.concatenate((Xpad.reshape(1, 20), Ypad.reshape(1, 20)), axis=0)

        h,w,c = im_size
        im_size = torch.FloatTensor([[h,w,c]])
        
        coordinate = torch.FloatTensor(point_coords).view(1, 2, 20)
        #target_points_norm = PointsToUnitCoords(point_coords, im_size)
        target_points_norm = PointsToUnitCoords(coordinate, im_size)

        h,w,c = warped_im_size
        warped_im_size = torch.FloatTensor([[h,w,c]])
        
        warped_points_aff_norm = self.pointTnf.affPointTnf(self.theta_identity, target_points_norm)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm, warped_im_size)
        
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords, warped_points_aff
 
    def pack_corr(self, a, warp_a, warp_b): 
        corr = np.zeros((20, 5))
        for i in range(len(a.numpy()[0])):
            if a[0][i] >= 0:
                corr[i][0] = warp_a.numpy()[0][0][i]
                corr[i][1] = warp_a.numpy()[0][1][i]
                corr[i][2] = warp_b.numpy()[0][0][i]
                corr[i][3] = warp_b.numpy()[0][1][i]
                corr[i][4] = 1
        corr = torch.FloatTensor(corr).view(20,5)
        return corr
