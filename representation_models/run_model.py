import torch
from loguru import logger
from torch import nn
from torchvision import models


class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Encoder:

    def __init__(self, params):

        self.params = params

        if params['model'] == 'VICReg':
            if params['architecture'] == 'ResNet':
                self.model = models.resnet50(pretrained=False)
            if params['architecture'] == 'AlexNet':
                self.model = models.alexnet(pretrained=False)
            if params['layer'] == 'avgpool':
                if params['architecture'] == 'ResNet':
                    self.model.fc = Identity()
                if params['architecture'] == 'AlexNet':
                    self.model.classifier = Identity()
            encoder_state_dict = torch.load(params['representation_model_path'],
                                            map_location=torch.device('cpu'))
            self.model.load_state_dict(encoder_state_dict['model_state_dict'])

        if params['model'] == 'BYOL':
            if params['architecture'] == 'ResNet':
                self.model = models.resnet50(pretrained=False)
            if params['architecture'] == 'AlexNet':
                self.model = models.alexnet(pretrained=False)

            trained_weights_path = params['representation_model_path']
            logger.critical("Load trained BYOL weights from {}".format(trained_weights_path))
            encoder_state_dict = torch.load(trained_weights_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(encoder_state_dict['model_state_dict'])

            if params['layer'] == 'avgpool':
                if params['architecture'] == 'ResNet':
                    self.model.fc = Identity()
                if params['architecture'] == 'AlexNet':
                    self.model.classifier = Identity()

        if params['model'] == 'ImageNet':
            if params['architecture'] == 'ResNet':
                self.model = models.resnet50(pretrained=True)
                logger.critical("Load imageNet model")
                self.model.fc = Identity()
            if params['architecture'] == 'AlexNet':
                self.model = models.alexnet(pretrained=False)
                self.model.classifier = Identity()
        if params['model'] == 'SIMClr':
            raise NotImplementedError
        if params['model'] == 'VICReg':
            raise NotImplementedError

        if params['eval'] == 1:
            self.model.eval()

    def encode(self, x):
        return self.model(x.reshape(1, 3, self.params['img_size'], self.params['img_size']))
