"""
@Origin : data.py by Yue Wang
@Contact: yuewangx@mit.edu
@Time: 2018/10/13 6:21 PM

modified by {Sanghyeok Lee, Sihyeon Kim}
@Contact: {cat0626, sh_bs15}@korea.ac.kr
@File: data.py
@Time: 2021.09.30
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

from PointWOLF import PointWOLF


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    # print("all_data",all_data.shape)#(9840, 2048,3)(2468,2048,3)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    ###change
    def __init__(self, args, partition='train'): #the origin,test
    # def __init__(self, args, saliency, partition='train'): #add a new parameter saliency,train
    ###end
        self.data, self.label = load_data(partition)
        self.num_points = args.num_points
        self.partition = partition
        ###add
        # self.saliency = saliency #train
        ###end
        self.PointWOLF = PointWOLF(args) if args.PointWOLF else None
        
        self.AugTune = args.AugTune

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points] #(1024,3)
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
            
            if self.PointWOLF is not None:
                ###change
                # origin, pointcloud = self.PointWOLF(pointcloud) #the origin
                origin, pointcloud = self.PointWOLF(pointcloud, self.saliency) #add a new parameter saliency
                ###end
                if self.AugTune:
                    #When AugTune used, we conduct CDA after AugTune.      
                    return origin, pointcloud, label
                
            pointcloud = translate_pointcloud(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
###modelnet40c
def normalize_pointcloud(pointcloud):
    pointcloud -= pointcloud.mean(0)
    d = ((pointcloud**2).sum(-1)**(1./2)).max()
    pointcloud /= d
    return pointcloud
def load_datac(data_root, label_root):
    point_set = np.load(data_root, allow_pickle=True)
    label = np.load(label_root, allow_pickle=True)
    return point_set, label

class ModelNetDataLoaderC(Dataset):
    def __init__(self, data_root, label_root, num_points, use_normals=False, partition='test'):
        assert partition in ['train', 'test']
        self.data, self.label = load_datac(data_root, label_root)
        self.num_points = num_points
        self.partition = partition
        self.use_normals = use_normals

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3] and its label as a scalar."""
        pc = self.data[item][:, :3]
        # print("pcc:",pc.shape)
        label = self.label[item]
        # print("labell:",label[0].shape)
        if self.use_normals:
            # pc = normalize_points_np(pc)
            pc[:, 0:3] = normalize_pointcloud(pc[:, 0:3])
        return pc, label[0]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
