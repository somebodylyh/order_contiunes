"""
generate_data.py — 用 FLATeacher 提取 wikitext hidden states，缓存到磁盘。

缓存结构（{cache_path}/{hash}/）:
  hidden.pt   Tensor [n_train+n_test, L, D]  原始顺序 hidden state
  perm.pt     Tensor [n_train+n_test, L]     每条序列的随机打乱置换
  meta.json   {"D", "L", "n_train", "n_test", "model_name", "layer_idx"}

三个实验 teacher config 完全相同 → 同一 cache key → 共享缓存，无需重复生成。

Usage:
  python -m gla_exp.generate_data --config gla_exp/configs/exp001_ar_noshuffle.yaml
  python -m gla_exp.generate_data --config gla_exp/configs/exp001_ar_noshuffle.yaml --force
"""
import sys, os, json, argparse, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from gla_exp.exp_config import load_config, teacher_cache_key, TeacherConfig
from gla_exp.teachers import FLATeacher


def get_cache_dir(cfg: TeacherConfig) -> str:
    return os.path.join(cfg.cache_path, teacher_cache_key(cfg))


def cache_exists(cache_dir: str) -> bool:
    return all(os.path.exists(os.path.join(cache_dir, f))
               for f in ["hidden.pt", "perm.pt", "meta.json"])


def _build_token_pools(teacher_cfg, teacher):
    """从 wikitext 构建 train/test token chunk 池，用于真实文本 hidden state 提取。

    dataset_name 支持:
      "wikitext"     → wikitext-2-raw-v1   (~85k train chunks)
      "wikitext-103" → wikitext-103-raw-v1 (~4.2M train chunks，推荐）
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[data] Teacher D={teacher.d_hidden}")
    print(f"[data] Loading tokenizer: {teacher_cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_cfg.model_name)

    if "103" in teacher_cfg.dataset_name:
        hf_name = "wikitext-103-raw-v1"
    else:
        hf_name = "wikitext-2-raw-v1"
    print(f"[data] Loading {hf_name}...")
    dataset = load_dataset("wikitext", hf_name)

    seq_len = teacher_cfg.seq_len

    def split_to_chunks(split_name):
        # 逐段 encode，避免一次性把 540M 字符拼成大字符串导致 OOM
        texts = [t for t in dataset[split_name]["text"] if t.strip()]
        chunks = []
        for txt in texts:
            ids = tokenizer.encode(txt, add_special_tokens=False)
            if ids:
                chunks.append(torch.tensor(ids, dtype=torch.long))
        t = torch.cat(chunks)
        n = len(t) // seq_len
        return t[:n * seq_len].view(n, seq_len)

    train_pool = split_to_chunks("train")
    val_pool   = split_to_chunks("validation")
    test_pool  = split_to_chunks("test")
    # merge validation+test so we have enough chunks for 10k test samples
    full_test_pool = torch.cat([val_pool, test_pool], dim=0)

    print(f"[data] Train pool: {train_pool.shape[0]} chunks  "
          f"Test pool: {full_test_pool.shape[0]} chunks  (seq_len={seq_len})")
    return train_pool, full_test_pool


def _fill_samples(teacher, token_pool, hidden_buf, perm_buf, batch_size, device, desc):
    """从 token_pool 随机采样（有放回），将 hidden states 填入预分配 buffer。"""
    n_samples = hidden_buf.shape[0]
    n_batches = math.ceil(n_samples / batch_size)
    n_pool    = token_pool.shape[0]

    print(f"[generate] 提取 {desc} {n_samples} 样本 ({n_batches} batches × ~{batch_size})...")
    offset = 0
    for i in range(n_batches):
        B         = min(batch_size, n_samples - offset)
        idx       = torch.randint(0, n_pool, (B,))
        input_ids = token_pool[idx].to(device)
        result    = teacher.extract(input_ids)
        hidden_buf[offset:offset + B].copy_(result["hidden"].cpu())
        perm_buf[offset:offset + B].copy_(result["perm"].cpu())
        offset += B
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_batches}", flush=True)


def generate_and_cache(teacher_cfg: TeacherConfig, force: bool = False) -> str:
    """生成并缓存 hidden states。返回 cache_dir 路径。"""
    cache_dir = get_cache_dir(teacher_cfg)

    if not force and cache_exists(cache_dir):
        print(f"[cache] Found: {cache_dir}")
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[generate] Loading teacher: {teacher_cfg.model_name}")
    teacher = FLATeacher(teacher_cfg).to(device)
    D = teacher.d_hidden

    L, B       = teacher_cfg.seq_len, teacher_cfg.extract_batch_size
    n_train, n_test = teacher_cfg.n_train, teacher_cfg.n_test
    N = n_train + n_test

    train_pool, test_pool = _build_token_pools(teacher_cfg, teacher)

    # 一次性预分配整个大 tensor，train/test 写入不同 slice，避免 cat 时双倍内存
    print(f"[generate] 预分配 hidden [{N}, {L}, {D}] ({N*L*D*4/1e9:.1f} GB)...")
    hidden_all = torch.zeros(N, L, D,  dtype=torch.float32)
    perm_all   = torch.zeros(N, L,     dtype=torch.long)

    _fill_samples(teacher, train_pool, hidden_all[:n_train], perm_all[:n_train], B, device, "train")
    _fill_samples(teacher, test_pool,  hidden_all[n_train:], perm_all[n_train:], B, device, "test")

    print(f"[generate] Saving to disk...")
    torch.save(hidden_all, os.path.join(cache_dir, "hidden.pt"))
    torch.save(perm_all,   os.path.join(cache_dir, "perm.pt"))
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({
            "D":          D,
            "L":          L,
            "n_train":    n_train,
            "n_test":     n_test,
            "model_name": teacher_cfg.model_name,
            "layer_idx":  teacher_cfg.layer_idx,
        }, f, indent=2)

    print(f"[generate] Done: {n_train} train + {n_test} test  shape=[{N}, {L}, {D}]")
    return cache_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t_cfg, _, _ = load_config(args.config)
    generate_and_cache(t_cfg, force=args.force)
