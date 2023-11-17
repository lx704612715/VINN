'''
modified Phil Wang's code
url: https://github.com/lucidrains/byol-pytorch
'''
import os

import torch
import datetime
from loguru import logger
from torchvision import models
from torchvision import transforms as T
from torch.utils.data import DataLoader

import tqdm
import sys
import wandb
from byol_pytorch import BYOL
from spock import spock, SpockBuilder


@spock
class TrainingConfig:
    # experiment parameters
    run_name: str
    root_dir: str
    folder: str
    save_dir: str
    extension: str
    dataset: str
    # neural network parameters
    img_size: int
    hidden_layer: str
    # training parameters
    lr: float
    gpu: int
    epochs: int
    wandb: int
    batch_size: int
    bc_model: str
    pretrained: int
    bc_model = None
    train_representation: int


if __name__ == '__main__':
    params = SpockBuilder(TrainingConfig).generate()
    params = params.TrainingConfig.__dict__

    curt_time = datetime.datetime.now()
    time_str = "_Time" + str(curt_time.minute) + str(curt_time.hour) + "_Day" + str(curt_time.day) + str(curt_time.month)

    if params["wandb"] == 1:
        wandb.init(project='BYOL', config=params)
        wandb.run.name = time_str + str(params["run_name"])

    sys.path.append(params['root_dir'])
    sys.path.append(params['root_dir'] + 'dataloaders')
    from dataloaders.CustomDataset import CustomDataset

    customAug = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6, 1.0)),
                           T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8, .8, .8, .2)]), p=.3),
                           T.RandomGrayscale(p=0.2),
                           T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                           T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                       std=torch.tensor([0.229, 0.224, 0.225]))])

    img_data = CustomDataset(params, None)

    if params['pretrained'] == 1:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(pretrained=False)

    if params['gpu'] == 1:
        device = torch.device('cuda')
        model = model.to(device)
        dataLoader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True, pin_memory=True, num_workers=8)
    else:
        dataLoader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True)

    learner = BYOL(
        model,
        image_size=params['img_size'],
        hidden_layer=params['hidden_layer'],
        augment_fn=customAug
    )

    optimizer = torch.optim.Adam(learner.parameters(), lr=params['lr'])
    epochs = params['epochs']
    best_loss = torch.inf

    # export model to the dir with the date as file name
    os.makedirs(params['save_dir'], exist_ok=True)
    save_dir = params["save_dir"] + params['run_name'] + time_str + "/"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm.tqdm(range(epochs), leave=False):
        epoch_loss = 0
        for i, data in enumerate(dataLoader, 0):
            img_tensor = data[0]
            if params['gpu'] == 1:
                loss = learner(img_tensor.float().to(device))
            else:
                loss = learner(img_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
            epoch_loss += loss.item() * img_tensor.shape[0]

        logger.info('train loss {}'.format(epoch_loss / len(img_data)))
        export_path = save_dir + "epoch_" + str(epoch) + "_"

        if best_loss < epoch_loss and epoch >= 20:
            torch.save({'model_state_dict': model.state_dict()}, export_path + "best_model.pt")

        if params['wandb'] == 1:
            wandb.log({'train loss': epoch_loss / len(img_data)})

        if epoch % 20 == 0:
            torch.save({'model_state_dict': model.state_dict()}, export_path + '.pt')
