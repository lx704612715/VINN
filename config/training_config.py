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
