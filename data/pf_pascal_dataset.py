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
    
class PFPascalDataset(Dataset):
    
    """
        Description:
            Proposal Flow image pair dataset

        Args:
            csv_file (string): Path to the csv file with image names and transformations.
            dataset_path (string): Directory with the images.
            output_size (2-tuple): Desired output size
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
    """

    def __init__(self, 
                 csv_file, 
                 dataset_path, 
                 dataset_size=None,
                 output_size=(240,240), 
                 transform=None, 
                 category=None, 
                 random_crop=True):

        self.category_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                               'sheep', 'sofa', 'train', 'tvmonitor']
        self.random_crop = random_crop
        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        if dataset_size is not None:
            dataset_size = min(dataset_size, len(self.pairs))
            self.pairs = self.pairs.iloc[0:dataset_size,:]
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

        flipA = False
        flipB = False
        if self.flip[idx] == 1:
            flipB = True
        elif self.flip[idx] == 2:
            flipA = True
        elif self.flip[idx] == 3:
            flipA = True
            flipB = True

        image_A, im_size_A, boundary_A = self.get_image(self.img_A_names, idx, flip=flipA)
        image_B, im_size_B, boundary_B = self.get_image(self.img_B_names, idx, flip=flipB)

        # get pre-processed point coords
        point_A_coords, warped_point_A_coords = self.get_points(self.point_A_coords, idx, flipA, im_size_A, (240,240,3), boundary_A)
        point_B_coords, warped_point_B_coords = self.get_points(self.point_B_coords, idx, flipB, im_size_B, (240,240,3), boundary_B)
        
        correspondence = self.pack_corr(point_A_coords, point_B_coords, warped_point_A_coords, warped_point_B_coords)
        #    if torch.sum(torch.sum(correspondence,1),0).numpy()[0] == 0:
        #        token = True
        #    else:
        #        token = False

        # category: class of pf-pascal, will be the index of class list plus 1
        image_category = self.category[idx]
        
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

        if self.random_crop:
            h,w,c = image.shape
            top = np.random.randint(h/4)
            bottom = int(3*h/4+np.random.randint(h/4))
            left = np.random.randint(w/4)
            right = int(3*w/4+np.random.randint(w/4))
            boundary = (top, bottom, left, right)
            image = image[top:bottom, left:right, :]
        
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
        
        return (image, im_size, boundary)
    
    def get_points(self, point_coords_list, idx, flip, im_size, warped_im_size, boundary):
        X = np.fromstring(point_coords_list.iloc[idx,0], sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx,1], sep=';')

        top, bottom, left, right = boundary
        if self.random_crop:
            X = X - left
            Y = Y - top

        ind = []
        for i in range(len(X)):
            if X[i] < 0 or X[i] >= (right-left) or Y[i] < 0 or Y[i] >= (bottom-top):
                ind.append(i)

        if flip:
            X = im_size[1] - X
        Xpad = -np.ones(20); Xpad[:len(X)] = X
        Ypad = -np.ones(20); Ypad[:len(X)] = Y
        
        if len(ind) != 0:
            for i in ind:
                Xpad[i] = -1
                Ypad[i] = -1
        point_coords = np.concatenate((Xpad.reshape(1, 20), Ypad.reshape(1, 20)), axis=0)

        h,w,c = im_size
        im_size = torch.FloatTensor([[h,w,c]])
        
        coordinate = torch.FloatTensor(point_coords).view(1, 2, 20)
        target_points_norm = PointsToUnitCoords(coordinate, im_size)

        h,w,c = warped_im_size
        warped_im_size = torch.FloatTensor([[h,w,c]])
        
        warped_points_aff_norm = self.pointTnf.affPointTnf(self.theta_identity, target_points_norm)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm, warped_im_size)
        
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords, warped_points_aff
 
    def pack_corr(self, a, b, warp_a, warp_b): 
        corr = np.zeros((20, 5))
        for i in range(len(a.numpy()[0])):
            if a[0][i] >= 0 and a[0][i] < self.out_w \
               and a[1][i] >= 0 and a[1][i] < self.out_h \
               and b[0][i] >= 0 and b[0][i] < self.out_w \
               and b[1][i] >= 0 and b[1][i] < self.out_h:
                corr[i][0] = warp_a.numpy()[0][0][i]
                corr[i][1] = warp_a.numpy()[0][1][i]
                corr[i][2] = warp_b.numpy()[0][0][i]
                corr[i][3] = warp_b.numpy()[0][1][i]
                corr[i][4] = 1
        corr = torch.FloatTensor(corr).view(20,5)
        return corr
