"""
generate_hspace_memmap.py — 用 ContinuousHSpaceTeacher 生成 h-space AR 序列，
保存为 baseline_continuous/disk_dataset.py 期望的 numpy memmap 格式。

Usage:
    python -m gla_exp.generate_hspace_memmap \\
        --data_dir baseline_continuous/data_hspace \\
        --sigma 0.3 --seq_len 32 \\
        --n_train 100000 --n_val 10000 --n_test 10000 \\
        --batch_size 64
"""
import sys, os, argparse, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gla_exp.exp_config import TeacherConfig
from gla_exp.teachers import ContinuousHSpaceTeacher


def fill_split(teacher, n_samples, seq_len, sigma, batch_size, device,
               vec_mmap, init_mmap, split_name):
    """批量生成序列，写入预分配的 memmap。"""
    n_batches = math.ceil(n_samples / batch_size)
    print(f"[{split_name}] {n_samples} samples × L={seq_len}  ({n_batches} batches)")
    offset = 0
    for i in range(n_batches):
        B   = min(batch_size, n_samples - offset)
        seq = teacher.generate_sequence(B=B, L=seq_len, sigma=sigma, device=device)
        # seq: [B, L, D]  (float32 on GPU)
        arr = seq.cpu().numpy()          # [B, L, D]
        vec_mmap [offset:offset + B] = arr          # 完整序列
        init_mmap[offset:offset + B] = arr[:, :1]   # h_0 as init
        offset += B
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_batches}", flush=True)
    vec_mmap.flush()
    init_mmap.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True,
                        help="输出目录，例如 baseline_continuous/data_hspace")
    parser.add_argument("--model_name", default="fla-hub/gla-340M-15B")
    parser.add_argument("--layer_idx",  type=int,   default=3)
    parser.add_argument("--sigma",      type=float, default=0.3)
    parser.add_argument("--seq_len",    type=int,   default=32)
    parser.add_argument("--n_train",    type=int,   default=100000)
    parser.add_argument("--n_val",      type=int,   default=10000)
    parser.add_argument("--n_test",     type=int,   default=10000)
    parser.add_argument("--batch_size", type=int,   default=64)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TeacherConfig(model_name=args.model_name, layer_idx=args.layer_idx)
    teacher = ContinuousHSpaceTeacher(cfg).to(device)
    D   = teacher.d_hidden
    L   = args.seq_len
    sig = args.sigma

    print(f"[config] D={D}, L={L}, sigma={sig}")
    print(f"[config] train={args.n_train}, val={args.n_val}, test={args.n_test}")

    splits = [
        ("train", args.n_train),
        ("val",   args.n_val),
        ("test",  args.n_test),
    ]

    for split, n in splits:
        vec_path  = os.path.join(args.data_dir, f"{split}_vectors.npy")
        init_path = os.path.join(args.data_dir, f"{split}_init_vectors.npy")
        vec_mmap  = np.memmap(vec_path,  dtype="float32", mode="w+", shape=(n, L, D))
        init_mmap = np.memmap(init_path, dtype="float32", mode="w+", shape=(n, 1, D))
        fill_split(teacher, n, L, sig, args.batch_size, device,
                   vec_mmap, init_mmap, split)
        print(f"[{split}] saved → {vec_path}")

    # 保存 data_config.pt（disk_dataset.py 需要）
    config = {
        "seq_length":    L,
        "vector_dim":    D,
        "num_init":      1,
        "train_samples": args.n_train,
        "val_samples":   args.n_val,
        "test_samples":  args.n_test,
        "sigma":         sig,
        "model_name":    args.model_name,
        "layer_idx":     args.layer_idx,
    }
    torch.save(config, os.path.join(args.data_dir, "data_config.pt"))
    print(f"[done] data_config.pt saved → {args.data_dir}")


if __name__ == "__main__":
    main()
