import torch
import torch.nn as nn 
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd 
import numpy as np 
from PIL import Image


class clf_dataset(Dataset):  
    def __init__(self, root_path, csv_path, transforms): 
        self.data_file = pd.read_csv(csv_path)
        self.root_path = root_path 
        self.tforms = transforms       

    def __len__(self):
        data_size = self.data_file.shape[0]
        return data_size 

    def __getitem__(self, idx):
        img_path = self.data_file.iloc[idx]['fp']
        cn = self.data_file.iloc[idx]['cn']
        cid = self.data_file.iloc[idx]['cid']

        # print("img path: ", img_path) 
        img = Image.open(img_path) 
        img_tfrmd = self.tforms(img) 

        output = {'X': img_tfrmd,
                  'Y': cid       }   

        return output


def main():
    root_path = './assignment1_data'
    csv_path = './data_files/train.csv'

    tform = transforms.Compose([                  
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = clf_dataset(root_path, csv_path, tform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)


    for id, data in enumerate(dataloader):
        x = data['X']
        y = data['Y']

        print("X : ", x.shape, " Y : ", y) 


if __name__ == "__main__":
    main()    