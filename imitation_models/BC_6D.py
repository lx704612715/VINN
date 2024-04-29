import os
import sys
import wandb
import torch
import datetime
import torch.nn.functional as F

from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms as T


class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TranslationModel(nn.Module):
    def __init__(self, input_dim):
        super(TranslationModel, self).__init__()
        self.f1 = nn.Linear(input_dim, input_dim)
        self.f2 = nn.Linear(input_dim, 1024)
        self.f3 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        return x


class RotationModel(nn.Module):
    def __init__(self, input_dim):
        super(RotationModel, self).__init__()
        self.f1 = nn.Linear(input_dim, input_dim)
        self.f2 = nn.Linear(input_dim, 1024)
        self.f3 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        return x


class GripperModel(nn.Module):
    def __init__(self, input_dim):
        super(GripperModel, self).__init__()
        self.f1 = nn.Linear(input_dim, 4)

    def forward(self, x):
        return self.f1(x)


class BC_Full:
    def __init__(self, params):
        self.params = params
        self.augment = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6, 1.0)),
                                  T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8, .8, .8, .2)]), p=.3),
                                  T.RandomGrayscale(p=0.2),
                                  T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                                  T.Normalize(
                                      mean=torch.tensor([0.485, 0.456, 0.406]),
                                      std=torch.tensor([0.229, 0.224, 0.225]))])

        sys.path.append(params['root_dir'] + 'representation_models')
        sys.path.append(params['root_dir'] + 'dataloaders')
        from run_model import Encoder
        from CustomDataset import CustomDataset

        if self.params['train_representation'] == 1:
            encoder = None
        else:
            encoder = Encoder(params)

        if self.params['wandb'] == 1:
            wandb.init(project='CustomDataset BC FULL')
            wandb.run.name = 'CustomDataset BC FULL'

        self.min_val_loss = float('inf')
        self.translation_loss_train = 0
        self.rotation_loss_train = 0
        self.translation_loss_val = 0
        self.rotation_loss_val = 0

        self.params['folder'] = self.params['train_dir']
        self.orig_img_data_train = CustomDataset(self.params, encoder)
        self.params['folder'] = self.params['val_dir']
        self.img_data_val = CustomDataset(self.params, encoder)

        self.dataLoader_train = DataLoader(self.orig_img_data_train, batch_size=self.params['batch_size'], shuffle=True,
                                           pin_memory=True)
        self.dataLoader_val = DataLoader(self.img_data_val, batch_size=self.params['batch_size'], shuffle=True,
                                         pin_memory=True)

        if self.params['gpu'] == 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if self.params['pretrain_encoder'] == 1:
            self.resnet = models.resnet50(pretrained=True).to(self.device)
        else:
            self.resnet = models.resnet50(pretrained=False).to(self.device)

        if params['architecture'] == 'ResNet':
            self.translation_model = TranslationModel(2048 * (self.params['t'] + 1)).to(self.device)
            self.rotation_model = RotationModel(2048 * (self.params['t'] + 1)).to(self.device)
        if params['architecture'] == 'AlexNet':
            self.translation_model = TranslationModel(9216 * (self.params['t'] + 1)).to(self.device)
            self.rotation_model = RotationModel(9216 * (self.params['t'] + 1)).to(self.device)

        curt_time = datetime.datetime.now()
        self.time_str = "_Time" + str(curt_time.minute) + str(curt_time.hour) + "_Day" + str(curt_time.day) + str(curt_time.month)

        os.makedirs(params['save_dir'], exist_ok=True)
        self.save_dir = params["save_dir"] + params['run_name'] + self.time_str + "/"
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        for epoch in tqdm(range(self.params['epochs'])):
            epoch_translation_loss_train = 0
            epoch_orientation_loss_train = 0

            for i, data in enumerate(self.dataLoader_train, 0):
                self.optimizer.zero_grad()

                if self.params['train_representation'] == 1:
                    image, translation, rotation = data
                    representation = self.resnet(self.augment(image).float().to(self.device))
                else:
                    representation, translation, rotation = data

                pred_translation = self.translation_model(representation.float().to(self.device))
                pred_rotation = self.rotation_model(representation.float().to(self.device))

                translation_loss = self.mseLoss(pred_translation, translation.float().to(self.device))
                rotation_loss = self.mseLoss(pred_rotation, rotation.float().to(self.device))

                epoch_translation_loss_train += translation_loss.item() * translation.shape[0]
                epoch_orientation_loss_train += rotation_loss.item() * translation.shape[0]
                loss = translation_loss + rotation_loss

                loss.backward()
                self.optimizer.step()

            epoch_translation_loss_train /= len(self.orig_img_data_train)
            epoch_orientation_loss_train /= len(self.orig_img_data_train)

            # running evaluation
            epoch_translation_loss_val, epoch_orientation_loss_val = self.val()

            if self.params['wandb'] == 1:
                self.wandb_publish(epoch_translation_loss_train, epoch_orientation_loss_train,
                                   epoch_translation_loss_val, epoch_orientation_loss_val)
            if epoch % 10 == 0:
                self.save_model(epoch)
            if epoch >= 20 and self.min_val_loss < epoch_translation_loss_val:
                self.save_model(epoch, name="best_model")
                self.min_val_loss = epoch_translation_loss_val

    def val(self):
        epoch_translation_loss_val = 0
        epoch_orientation_loss_val = 0

        for i, data in enumerate(self.dataLoader_val, 0):
            image, translation, rotation = data
            if self.params['train_representation'] == 1:
                image, translation, rotation = data
                representation = self.resnet(self.augment(image).float().to(self.device))
            else:
                representation, translation, rotation = data

            pred_translation = self.translation_model(representation.float().to(self.device))
            pred_rotation = self.rotation_model(representation.float().to(self.device))
            translation_loss = self.mseLoss(pred_translation, translation.float().to(self.device))
            rotation_loss = self.mseLoss(pred_rotation, rotation.float().to(self.device))

            epoch_translation_loss_val += translation_loss.item() * image.shape[0]
            epoch_orientation_loss_val += rotation_loss.item() * image.shape[0]

        epoch_translation_loss_val /= len(self.img_data_val)
        epoch_orientation_loss_val /= len(self.img_data_val)

        return epoch_orientation_loss_val, epoch_orientation_loss_val

    def training(self):
        losses = []
        if self.params['gpu'] == 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if self.params['pretrain_encoder'] == 1:
            self.resnet = models.resnet50(pretrained=True).to(self.device)
        else:
            self.resnet = models.resnet50(pretrained=False).to(self.device)

        if self.params['architecture'] == 'ResNet':
            self.translation_model = TranslationModel(2048 * (self.params['t'] + 1)).to(self.device)
            self.rotation_model = RotationModel(2048 * (self.params['t'] + 1)).to(self.device)
        if self.params['architecture'] == 'AlexNet':
            self.translation_model = TranslationModel(9216 * (self.params['t'] + 1)).to(self.device)
            self.rotation_model = RotationModel(9216 * (self.params['t'] + 1)).to(self.device)

        if self.params['train_representation'] == 1:
            parameters = list(self.resnet.parameters()) + list(self.translation_model.parameters()) + \
                         list(self.rotation_model.parameters())
        else:
            parameters = list(self.translation_model.parameters()) + list(self.rotation_model.parameters())

        self.resnet.fc = Identity()
        self.optimizer = torch.optim.Adam(parameters, lr=self.params['lr'])

        self.dataLoader_train = DataLoader(self.orig_img_data_train, batch_size=self.params['batch_size'], shuffle=True,
                                           pin_memory=True)
        self.mseLoss = nn.MSELoss()
        self.ceLoss = nn.CrossEntropyLoss()

        self.train()
        print(self.translation_loss_val)
        losses.append(self.translation_loss_val)

        return losses

    def wandb_publish(self, trans_loss_train, rot_loss_train, trans_loss_val, rot_loss_val):
        wandb.log({'translation train': trans_loss_train, 'rotation train': rot_loss_train,
                   'translation val': trans_loss_val, 'rotation val': rot_loss_val})

    def save_model(self, epoch, name=None):
        if name is None:
            weights_name = self.params["run_name"] + '_' + str(epoch) + '.pt'
        else:
            weights_name = name + self.params["run_name"] + '_' + str(epoch) + '.pt'

        torch.save({'model_state_dict': self.translation_model.state_dict()}, self.save_dir + 'translation_m' + weights_name)
        torch.save({'model_state_dict': self.rotation_model.state_dict()}, self.save_dir + 'rotation_m' + weights_name)
        if self.params["train_representation"] == 1:
            torch.save({'model_state_dict': self.resnet.state_dict()}, self.save_dir + 'trained_resnet_' + weights_name)
