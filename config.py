import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # === training ===
    batch_size: int = 16
    total_training_steps: int = 50_000
    evaluation_frequency: int = 100
    checkpoint_save_frequency: int = 10_000
    evaluation_loops: int = 10

    # === sequence ===
    input_sequence_length: int = 1024
    max_sequence_length: int = 2048

    # === model ===
    embedding_dim: int = 384
    hidden_dim: int = 1536
    num_attention_heads: int = 6
    layer_count: int = 20
    rope_theta: float = 1_000_000.0
    vocab_size: int = 50257

    # === optimization ===
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 1_000

    # === system ===
    device_type: str = "cuda"
    random_seed_value: int = 1337
    autocast_dtype: torch.dtype = torch.bfloat16
