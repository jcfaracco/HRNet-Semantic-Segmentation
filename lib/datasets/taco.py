import os
import torch
import numpy as np
import torchvision
from typing import Dict, Optional, Tuple, Any
from PIL import Image
import csv
from pycocotools import coco
import json
import cv2

from .base_dataset import BaseDataset


class TACO(BaseDataset):
    def __init__(self,
         root,
         ann_file="/TACO/data/annotations_0_",
         class_cfg="/TACO/detector/taco_config/map_4.csv",
         list_path='train',
         num_samples=None,
         num_classes=5,
         multi_scale=False, 
         flip=True, 
         ignore_label=-1,
         base_size=1080,
         crop_size=(720, 720),
         downsample_rate=1,
         scale_factor=16,
         center_crop_test=False,
         mean=[0.485, 0.456, 0.406], 
         std=[0.229, 0.224, 0.225],
    ):
        super().__init__(
            ignore_label,
            base_size,
            crop_size,
            downsample_rate,
            scale_factor,
            mean,
            std,
        )

        self.root = root
        self.ann_file = ann_file + list_path + '.json'

        self.split = list_path
        
        self.class_map = self._process_csv(class_cfg)

        self.num_classes = num_classes
        self.class_weights = None # torch.FloatTensor([0.2, 1, 1, 1, 1]).cuda() 

        self.multi_scale = multi_scale
        self.flip = flip

        self.coco = coco.COCO(self.ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _process_csv(csvpath: str) -> Dict:
        str2id = {}
        class2id = {}
        with open(csvpath) as f:
            reader = csv.reader(f)
            count = 1 # background is zero
            for i, row in enumerate(reader):
                if row[1] in str2id:
                    class2id[i] = str2id[row[1]]
                else:
                    str2id[row[1]] = count
                    class2id[i] = count
                    count += 1
        return class2id

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        anns_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(anns_ids)

        name = self.coco.loadImgs(img_id)[0]['file_name']
        impath = os.path.join(self.root, name)

        image = cv2.imread(impath, cv2.IMREAD_COLOR)
        size = image.shape[:2]

        label = np.zeros(size, dtype=np.int)
        for ann in anns:
            mask = self.coco.annToMask(ann).astype(np.bool)
            if mask.shape != label.shape:
                print('mask label shape mismatch', mask.shape, label.shape)
            else:
                label[mask] = self.class_map[ann['category_id']]

        if self.split == 'val':
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            label = cv2.resize(label, self.crop_size, 
                               interpolation=cv2.INTER_NEAREST)
            label = self.label_transform(label)

        elif self.split == 'testval':
            # evaluate model on val dataset
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            label = self.label_transform(label)
        else:
            image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)
                                
        return image.copy(), label.copy(), np.array(size), name

if __name__ == '__main__':

    taco_dir = "/home/jbragantini/Softwares/TACO"
    taco = TACO(taco_dir + "/data",
                taco_dir + "/data/annotations.json",
                taco_dir + "/detector/taco_config/map_4.csv")
  
