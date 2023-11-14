import json
import glob
import random
import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from PIL import Image, ImageFile
from collections import defaultdict

import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDataset(Dataset):

    def __init__(self, params, encoder, partial=None):
        self.params = params
        self.encoder = encoder

        self.img_tensors = []
        self.representations = []
        self.translation = []
        self.rotation = []
        self.gripper = []
        self.paths = []
        self.path_dict = defaultdict(list)
        self.frame_index = defaultdict(int)

        self.preprocess = T.Compose([T.ToTensor(),
                                    T.Resize((self.params['img_size'], self.params['img_size'])),
                                    T.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

        self.extract_data(partial)

    def extract_data(self, factor=None):
        folder_path = self.params['folder']
        annotation_file_path = folder_path + "/annotation_files.pkl"
        annotation_files = pd.read_pickle(open(annotation_file_path, "rb"))

        image_names_list = annotation_files["rgb_img_name"].to_list()
        random.shuffle(image_names_list)
        total = len(image_names_list)

        for img_name in image_names_list:
            img_path = os.path.join(folder_path+"/iH_rgb/", img_name)
            img = Image.open(img_path)
            img_tensor = self.preprocess(img)

            index = annotation_files["rgb_img_name"] == img_name
            action = annotation_files["ee_se3_ee"][index].to_numpy()[0]
            gripper_state = annotation_files["is_grasped"][index].to_numpy()
            self.translation.append(torch.FloatTensor(action[0:3]))
            self.rotation.append(torch.FloatTensor(action[3:6]))
            self.gripper.append(torch.FloatTensor(gripper_state))

            if self.params['representation'] == 1:
                self.img_tensors.append(img_tensor.detach())
            else:
                representation = self.encoder.encode(img_tensor)[0]
                self.representations.append(representation.detach())

    def __len__(self):
        return max(len(self.img_tensors), len(self.representations))

    def __getitem__(self, index):
        if self.params['representation'] == 1:
            if self.params['bc_model'] == 'BC_Full':
                return self.img_tensors[index], self.translation[index], self.rotation[index], self.gripper[index], self.paths[index]
            else:
                return self.img_tensors[index]
        else:
            return self.representations[index], self.translation[index], self.rotation[index], self.gripper[index], self.paths[index]
