from dataclasses import dataclass

@dataclass
class ModelConfig:
    pass

@dataclass
class TrainConfig:
    batch_size : int = 64
    
    log_every : int = 10
    save_every : int = 10
    save_to : str = "checkpoints"
    eval_every : int = 10
    sample_every : int = 10