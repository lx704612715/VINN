import sys
import numpy as np
import argparse
from spock import spock, SpockBuilder
from loguru import logger


@spock
class TrainingBCConfig:
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
    pretrained: int
    # BC configuration
    train_representation: int = 1
    t: int = 0
    model: str = "BYOL"
    bc_model: str
    train_dir: str
    val_dir: str
    representation_model_path: str
    layer: str = "avgpool"
    architecture: str = "ResNet"
    eval: int = 0
    pretrain_encoder: int = 1


if __name__ == '__main__':
    params = SpockBuilder(TrainingBCConfig).generate()
    params = params.TrainingBCConfig.__dict__

    sys.path.append(params['root_dir'] + 'imitation_models')
    from BC_6D import BC_Full
    bc = BC_Full(params)
    logger.critical("Start Training")
    bc.training()


