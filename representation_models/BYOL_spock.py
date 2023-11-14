'''
modified Phil Wang's code
url: https://github.com/lucidrains/byol-pytorch
'''


import torch
import datetime
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
    representation: int


if __name__ == '__main__':
    params = SpockBuilder(TrainingConfig).generate()
    params = params.TrainingConfig.__dict__
    params["representation"] = 1

    curt_time = datetime.datetime.now()
    time_str = "_" + str(curt_time.minute) + str(curt_time.hour) + "_" + str(curt_time.day) + str(curt_time.month)

    if params["wandb"] == 1:
        wandb.init(project='vinn', config=params)
        wandb.run.name = time_str + str(params["run_name"])

    sys.path.append(params['root_dir'])
    sys.path.append(params['root_dir'] + 'dataloaders')
    from dataloaders.PushDataset import PushDataset
    from dataloaders.HandleDataset import HandleDataset
    from dataloaders.CustomDataset import CustomDataset

    customAug = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6, 1.0)),
                           T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8, .8, .8, .2)]), p=.3),
                           T.RandomGrayscale(p=0.2),
                           T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                           T.Normalize(
                               mean=torch.tensor([0.485, 0.456, 0.406]),
                               std=torch.tensor([0.229, 0.224, 0.225]))])

    if params['dataset'] == 'HandleData':
        img_data = HandleDataset(params, None)
    if params['dataset'] == 'PushDataset' or params['dataset'] == 'StackDataset':
        img_data = PushDataset(params, None)
    if params['dataset'] == 'CustomDataset':
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

    for epoch in tqdm.tqdm(range(epochs), leave=False):
        epoch_loss = 0
        for i, data in enumerate(dataLoader, 0):
            if params['gpu'] == 1:
                loss = learner(data.float().to(device))
            else:
                loss = learner(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
            epoch_loss += loss.item() * data.shape[0]

        print('train loss {}'.format(epoch_loss / len(img_data)))
        if best_loss < epoch_loss:
            torch.save({'model_state_dict': model.state_dict()}, params['save_dir'] + "BYOL_" + str(epoch) + "best_model.pt")

        if params['wandb'] == 1:
            wandb.log({'train loss': epoch_loss / len(img_data)})

        if epoch % 20 == 0:
            torch.save({'model_state_dict': model.state_dict()}, params['save_dir']+'BYOL_' + str(epoch) + '_' +
                       params['extension'] + '_pretrained_' + str(params['pretrained']) + '.pt')
