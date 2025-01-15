""" load data from hd5f datasets which are also used for ACT and Diffusion Policy
"""
import os
from collections import defaultdict
import torch
import torchvision.transforms as T
import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import h5py
import numpy as np
from LfDusingEC.utils.utils_robotMath import *
from pytransform3d.rotations import norm_matrix
from pytransform3d.transformations import exponential_coordinates_from_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()
    with h5py.File(dataset_path, 'r') as root:
        base_ht_ee = root['observations']['base_ht_ee'][()]
        images = root['observations']['images']['front'][()]

        # convert base_ht_ee
        ee_se3_ee = np.zeros([base_ht_ee.shape[0], 6])  # translation, rotation
        for i in tqdm.tqdm(range(1, base_ht_ee.shape[0])):
            ee_ht_ee = TransInv(base_ht_ee[i-1]) @ base_ht_ee[i]
            ee_ht_ee[:3, :3] = norm_matrix(ee_ht_ee[:3, :3])
            # compute delta movement in EE frame in Lie-space (translation, rotation)
            try:
                tmp_ee_se3_ee = exponential_coordinates_from_transform(ee_ht_ee)
            except Exception as e:
                print(e)
                tmp_ee_se3_ee = np.zeros(6)
            ee_se3_ee[i][:3] = tmp_ee_se3_ee[3:]  # translation
            ee_se3_ee[i][3:] = tmp_ee_se3_ee[:3]  # rotation

    return images, ee_se3_ee


class HD5FDataset(Dataset):

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
        self.preprocess = T.Compose([T.ToTensor(), T.Resize((self.params['img_size'], self.params['img_size'])),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.extract_data(partial)

    def extract_data(self, factor=None):
        folder_path = self.params['folder']
        # load all episode data
        all_episodes_names = [name for name in os.listdir(folder_path) if
                              os.path.isfile(os.path.join(folder_path, name))]

        shapes = []
        aligned_data = dict()
        # first get all dataset shape
        for name in all_episodes_names:
            logger.info("Processing {}".format(name))
            aligned_data[name] = dict()
            dataset_path = os.path.join(folder_path, name)
            images, ee_se3_ee = load_hdf5(dataset_path=dataset_path)

            for i in range(images.shape[0]):
                img_tensor = self.preprocess(images[i])
                self.translation.append(torch.FloatTensor(ee_se3_ee[i][0:3]))
                self.rotation.append(torch.FloatTensor(ee_se3_ee[i][3:6]))

                if self.params['train_representation'] == 1:
                    self.img_tensors.append(img_tensor.detach())
                else:
                    representation = self.encoder.encode(img_tensor)[0]
                    self.representations.append(representation.detach())

        logger.debug("Data Loaded")

    def __len__(self):
        return max(len(self.img_tensors), len(self.representations))

    def __getitem__(self, index):
        if self.encoder is not None and self.params["train_representation"] == 0:
            return self.representations[index], self.translation[index], self.rotation[index]
        else:
            return self.img_tensors[index], self.translation[index], self.rotation[index]
