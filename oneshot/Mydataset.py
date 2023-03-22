import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
import random
from PIL import Image

class ISBICellDataloader(data.Dataset):
    def __init__(self, root=None, dataset_type='train', cross=None, K=None, transform1=None, transform2=None):
        self.root = root
        self.dataset_type = dataset_type
        self.transform1 = transform1
        self.transform2 = transform2

        self.data = sorted(os.listdir(self.root + "/Image"))
        self.label = sorted(os.listdir(self.root + "/Label"))
        self.data = np.array(self.data)
        self.label = np.array(self.label)
        
        idx = np.arange(0, len(self.data))
        idx_train = idx[np.fmod(idx, K) != cross].astype(np.int32) 
        idx_test = idx[np.fmod(idx, K) == cross].astype(np.int32) 

        if self.dataset_type=='train':
            self.datas = self.data[idx_train[1]]
            self.labels = self.label[idx_train[1]]
        
        else:
            self.datas = self.data[idx_test]
            self.labels = self.label[idx_test]
        
        self.data_path = []
        self.label_path = []
        if self.dataset_type=='train':
            self.data_num = sorted(os.listdir(self.root + "/Image/{}".format(self.datas)))
            self.label_num = sorted(os.listdir(self.root + "/Label/{}".format(self.labels)))
            self.data_path.append("{}/".format(self.datas) + self.data_num[1])
            self.label_path.append("{}/".format(self.labels) + self.label_num[1])
        else:
            for i in range(len(self.datas)):
                self.data_num = sorted(os.listdir(self.root + "/Image/{}".format(self.datas[i])))
                self.label_num = sorted(os.listdir(self.root + "/Label/{}".format(self.labels[i])))
                for j in range(len(self.data_num)):
                    self.data_path.append("{}/".format(self.datas[i]) + self.data_num[j])
                    self.label_path.append("{}/".format(self.labels[i]) + self.label_num[j])
            
         
        data_num_sample = sorted(os.listdir(self.root + "/Image/{}".format(self.data[idx_train[0]])))
        label_num_sample = sorted(os.listdir(self.root + "/Label/{}".format(self.label[idx_train[0]])))
        sample_image_all = Image.open(self.root + "/Image/" + self.data[idx_train[0]] + "/" + data_num_sample[0]).convert("RGB")
        sample_label_all = Image.open(self.root + "/Label/" + self.label[idx_train[0]] + "/" + label_num_sample[0]).convert("RGB")
        
        self.sample_image = sample_image_all.crop((0, 0, 64, 64))
        self.sample_label = sample_label_all.crop((0, 0, 64, 64))

    def __getitem__(self, index):
        # data
        if self.dataset_type=='train':
            image_name = self.root + "/Image/" + self.data_path[0]
            label_name = self.root + "/Label/" + self.label_path[0]
        else:
            image_name = self.root + "/Image/" + self.data_path[index]
            label_name = self.root + "/Label/" + self.label_path[index]
        image = Image.open(image_name).convert("RGB")
        label = Image.open(label_name).convert("RGB")
        label = np.array(label)

        label = np.where(label[:,:,0]>=150, 0, 1)
        label = Image.fromarray(np.uint8(label))

        sample_image = self.sample_image
        sample_label = torch.Tensor(np.array(self.sample_label))
        sample_label = np.where(sample_label[:,:,0]>=150, 0, 1).astype(np.float32)
        sample_label = Image.fromarray(np.uint8(sample_label))
        
        if self.transform1:
            image, label = self.transform1(image, label)
            
        if self.transform2:
            sample_image, sample_label = self.transform2(sample_image, sample_label)

        return image, label, sample_image, sample_label

    def __len__(self):
        return len(self.data_path)

