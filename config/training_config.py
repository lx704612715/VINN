from spock import spock


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

    t = 0
    model = "BYOL"
    bc_model = None
    train_dir = None
    val_dir = None
    test_dir = None
    representation_model_path = None
    layer = "avgpool"
    architecture = "ResNet"
    eval = 0
    temporal = 0
