"""YAML config for Linear Attention Teacher-Student experiments."""
import hashlib, json
from dataclasses import dataclass, asdict
import yaml


@dataclass
class TeacherConfig:
    model_name: str = "fla-hub/gla-340M-15B"
    layer_idx: int = 3          # 使用前 (layer_idx+1) 层；3 → 4层，5 → 6层
    seq_len: int = 32
    n_train: int = 100000       # 训练集样本数
    n_test: int = 10000         # 测试集样本数
    extract_batch_size: int = 64  # teacher forward 时的 GPU batch size
    dataset_name: str = "wikitext"
    cache_path: str = "data/teacher_cache"


@dataclass
class StudentConfig:
    type: str = "ar_noshuffle"    # ar_noshuffle | ar_shuffled | mdm_shuffled
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mask_ratio: float = 0.15
    chunk_size: int = 1           # 1=token-level shuffle; >1=chunk-based shuffle


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    epochs: int = 50
    batch_size: int = 256
    log_interval: int = 5
    warmup_ratio: float = 0.05    # fraction of total steps used for LR warmup
    ema_decay: float = 0.9999     # target EMA decay (adaptive warm-up applied)
    grad_clip: float = 1.0
    noise_scale: float = 0.0      # σ: 训练时对 hidden states 加 N(0,σ²) 噪声；loss 下界 = σ²


def load_config(yaml_path: str):
    """返回 (TeacherConfig, StudentConfig, TrainingConfig)。"""
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    teacher  = TeacherConfig(**d.get("teacher", {}))
    student  = StudentConfig(**d.get("student", {}))
    training = TrainingConfig(**d.get("training", {}))
    return teacher, student, training


def teacher_cache_key(cfg: TeacherConfig) -> str:
    """8-char MD5 over fields that define the cached data.
    三个实验的 teacher 相同 → 返回同一 hash → 共享缓存。"""
    key_fields = {
        "model_name":   cfg.model_name,
        "layer_idx":    cfg.layer_idx,
        "seq_len":      cfg.seq_len,
        "n_train":      cfg.n_train,
        "n_test":       cfg.n_test,
        "dataset_name": cfg.dataset_name,
    }
    return hashlib.md5(
        json.dumps(key_fields, sort_keys=True).encode()
    ).hexdigest()[:8]
