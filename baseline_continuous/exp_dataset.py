"""
exp_dataset.py — 从缓存的 hidden.pt/perm.pt 构建 Dataset。

hidden.pt 布局：前 n_train 条为训练集，后 n_test 条为测试集（由 meta.json 记录）。

student_type:
  "ar_noshuffle": input=hidden（原始顺序）
  "ar_shuffled":  打乱顺序，chunk_size=1 时 token-level，>1 时块间打乱
  "mdm_shuffled": 打乱顺序，chunk_size=1 时 token-level，>1 时块间打乱

块间打乱（chunk_size > 1）：
  将 L 个位置划分为 L//chunk_size 个连续块，随机打乱块的顺序，
  块内相对顺序不变。每个样本的打乱方式由全局样本 index 作种子，
  保证 train/test 评估结果可复现。
"""
import os, json
import torch
from torch.utils.data import Dataset, DataLoader


def _chunk_perm(global_idx: int, L: int, chunk_size: int) -> torch.Tensor:
    """生成块间打乱的 token-level 排列（以 global_idx 为种子，可复现）。"""
    g = torch.Generator()
    g.manual_seed(int(global_idx))
    n_chunks = L // chunk_size
    cp = torch.randperm(n_chunks, generator=g)           # [n_chunks]
    # expand: cp[i] 号块对应原始位置 [cp[i]*chunk_size ... cp[i]*chunk_size+chunk_size-1]
    token_perm = (cp.unsqueeze(1) * chunk_size +
                  torch.arange(chunk_size)).flatten()    # [L]
    return token_perm


class HiddenStateDataset(Dataset):
    def __init__(self, cache_dir: str, student_type: str,
                 split: str = "train", chunk_size: int = 1):
        hidden_path = os.path.join(cache_dir, "hidden.pt")
        perm_path   = os.path.join(cache_dir, "perm.pt")
        assert os.path.exists(hidden_path), \
            f"缓存不存在: {hidden_path}\n请先运行: python -m baseline_continuous.generate_data --config <your.yaml>"

        self.hidden = torch.load(hidden_path, mmap=True, weights_only=True)
        self.perm   = torch.load(perm_path,   mmap=True, weights_only=True)

        with open(os.path.join(cache_dir, "meta.json")) as f:
            meta = json.load(f)

        assert student_type in ("ar_noshuffle", "ar_shuffled", "mdm_shuffled"), \
            f"Unknown student_type: {student_type!r}"
        assert chunk_size >= 1, f"chunk_size 必须 >= 1，got {chunk_size}"
        self.student_type = student_type
        self.chunk_size   = chunk_size

        n_train = meta["n_train"]
        n_test  = meta["n_test"]
        if split == "train":
            self._idx = torch.arange(n_train)
        else:
            self._idx = torch.arange(n_train, n_train + n_test)

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, i):
        idx    = self._idx[i].item()
        hidden = self.hidden[idx].clone().float()   # [L, D]

        if self.student_type == "ar_noshuffle":
            return {"input": hidden}

        L = hidden.shape[0]
        if self.chunk_size == 1:
            # token-level shuffle：使用缓存中 teacher 生成的随机排列
            perm = self.perm[idx].clone().long()
            return {"input": hidden[perm]}
        else:
            # 块间打乱：以全局 sample index 为种子，保证 train/test 一致
            perm = _chunk_perm(idx, L, self.chunk_size)
            return {"input": hidden[perm]}


def create_dataloaders(cache_dir: str, student_type: str, batch_size: int,
                       chunk_size: int = 1, num_workers: int = 4):
    train_ds = HiddenStateDataset(cache_dir, student_type, "train", chunk_size)
    test_ds  = HiddenStateDataset(cache_dir, student_type, "test",  chunk_size)
    print(f"[dataset] {student_type} (chunk_size={chunk_size}): "
          f"train={len(train_ds)}, test={len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
