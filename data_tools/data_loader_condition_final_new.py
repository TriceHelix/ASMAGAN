#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: data_loader_modify.py
# Created Date: Saturday April 4th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 28th April 2020 10:42:50 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import os
import torch
import random
from PIL import Image
from pathlib import Path
from torch.utils import data
import torchvision.datasets as dsets
from torchvision import transforms as T
from data_tools.StyleResize import StyleResize
# from StyleResize import StyleResize

class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.content, self.style, self.label = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.content, self.style, self.label = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.content= self.content.cuda(non_blocking=True)
            self.style  = self.style.cuda(non_blocking=True)
            self.label  = self.label.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        content = self.content
        style   = self.style
        label   = self.label 
        self.preload()
        return content, style, label

class TotalDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self, content_image_dir,style_image_dir,
                    selectedContent,selectedStyle,
                    content_transform,style_transform,
                    subffix='jpg', random_seed=1234):
        """Initialize and preprocess the Artworks dataset and content dataset."""
        self.content_image_dir= content_image_dir
        self.style_image_dir  = style_image_dir
        self.content_transform= content_transform
        self.style_transform  = style_transform
        self.selectedContent  = selectedContent
        self.selectedStyle    = selectedStyle
        self.subffix            = subffix
        self.content_dataset    = []
        self.art_dataset        = []
        self.random_seed= random_seed
        self.preprocess()
        self.num_images = len(self.content_dataset)
        self.art_num    = len(self.art_dataset)

    def preprocess(self):
        """Preprocess the Artworks dataset."""
        print("processing content images...")
        for dir_item in self.selectedContent:
            join_path = Path(self.content_image_dir,dir_item.replace('/','_'))
            if join_path.exists():
                print("processing %s"%dir_item,end='\r')
                images = join_path.glob('*.%s'%(self.subffix))
                for item in images:
                    self.content_dataset.append(item)
            else:
                print("%s dir does not exist!"%dir_item,end='\r')
        label_index = 0
        print("processing style images...")
        for class_item in self.selectedStyle:
            images = Path(self.style_image_dir).glob('%s/*.%s'%(class_item, self.subffix))
            for item in images:
                self.art_dataset.append([item, label_index])
            label_index += 1
        random.seed(self.random_seed)
        random.shuffle(self.content_dataset)
        random.shuffle(self.art_dataset)
        # self.dataset = images
        print('Finished preprocessing the Art Works dataset, total image number: %d...'%len(self.art_dataset))
        print('Finished preprocessing the Content dataset, total image number: %d...'%len(self.content_dataset))

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename        = self.content_dataset[index]
        image           = Image.open(filename)
        content         = self.content_transform(image)
        art_index       = random.randint(0,self.art_num-1)
        filename,label  = self.art_dataset[art_index]
        image           = Image.open(filename)
        style           = self.style_transform(image)
        return content,style,label

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def getLoader(s_image_dir,c_image_dir, 
                style_selected_dir, content_selected_dir,
                crop_size=178, batch_size=16, num_workers=8, 
                colorJitterEnable=True, colorConfig={"brightness":0.05,"contrast":0.05,"saturation":0.05,"hue":0.05}):
    """Build and return a data loader."""
    s_transforms = []
    c_transforms = []
    
    s_transforms.append(StyleResize())
    # s_transforms.append(T.Resize(900))
    c_transforms.append(T.Resize(900))

    s_transforms.append(T.RandomCrop(crop_size,pad_if_needed=True,padding_mode='reflect'))
    c_transforms.append(T.RandomCrop(crop_size))

    s_transforms.append(T.RandomHorizontalFlip())
    c_transforms.append(T.RandomHorizontalFlip())
    
    s_transforms.append(T.RandomVerticalFlip())
    c_transforms.append(T.RandomVerticalFlip())

    if colorJitterEnable:
        if colorConfig is not None:
            print("Enable color jitter!")
            colorBrightness = colorConfig["brightness"]
            colorContrast   = colorConfig["contrast"]
            colorSaturation = colorConfig["saturation"]
            colorHue        = (-colorConfig["hue"],colorConfig["hue"])
            s_transforms.append(T.ColorJitter(brightness=colorBrightness,\
                                contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
            c_transforms.append(T.ColorJitter(brightness=colorBrightness,\
                                contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
    s_transforms.append(T.ToTensor())
    c_transforms.append(T.ToTensor())

    s_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    s_transforms = T.Compose(s_transforms)
    c_transforms = T.Compose(c_transforms)

    content_dataset = TotalDataset(c_image_dir,s_image_dir, content_selected_dir, style_selected_dir
                        , c_transforms,s_transforms)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=True,shuffle=True,num_workers=num_workers,pin_memory=True)
    prefetcher = data_prefetcher(content_data_loader)
    return prefetcher

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
