"""
generate_hspace_memmap.py — 用 ContinuousHSpaceTeacher 生成 h-space AR 序列，
可选 bottleneck 投影降维（因果结构在低维空间闭环），
保存为 baseline_continuous/disk_dataset.py 期望的 numpy memmap 格式。

Usage:
    # 不投影（1024维）
    python -m gla_exp.generate_hspace_memmap \\
        --data_dir baseline_continuous/data_hspace \\
        --sigma 0.3 --seq_len 32

    # Bottleneck 投影到 32 维
    python -m gla_exp.generate_hspace_memmap \\
        --data_dir baseline_continuous/data_hspace_d32 \\
        --sigma 0.3 --seq_len 32 --proj_dim 32
"""
import sys, os, argparse, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gla_exp.exp_config import TeacherConfig
from gla_exp.teachers import ContinuousHSpaceTeacher


def make_ortho_proj(d, D, seed=0):
    """Random orthonormal projection R^D -> R^d.  W @ W^T = I_d."""
    rng = np.random.RandomState(seed)
    G = rng.randn(D, d).astype(np.float32)
    Q, _ = np.linalg.qr(G)          # [D, d], columns orthonormal
    return Q[:, :d].T.copy()         # [d, D]


def fill_split(teacher, n_samples, seq_len, sigma, batch_size, device,
               vec_mmap, init_mmap, split_name, W_gpu=None,
               rescale_mode='none', rescale_alpha=1.0,
               noise_type='process'):
    """批量生成序列，写入预分配的 memmap。

    当 W_gpu 不为 None 时，使用 bottleneck 模式：
    投影在 generate_sequence 内部完成（因果结构在低维空间闭环）。
    """
    n_batches = math.ceil(n_samples / batch_size)
    print(f"[{split_name}] {n_samples} samples × L={seq_len}  ({n_batches} batches)")
    offset = 0
    for i in range(n_batches):
        B   = min(batch_size, n_samples - offset)
        seq = teacher.generate_sequence(B=B, L=seq_len, sigma=sigma,
                                        device=device, W=W_gpu,
                                        rescale_mode=rescale_mode,
                                        rescale_alpha=rescale_alpha,
                                        noise_type=noise_type)
        arr = seq.cpu().numpy()
        vec_mmap [offset:offset + B] = arr
        init_mmap[offset:offset + B] = arr[:, :1]
        offset += B
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_batches}", flush=True)
    vec_mmap.flush()
    init_mmap.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True,
                        help="输出目录，例如 baseline_continuous/data_hspace_d64")
    parser.add_argument("--model_name", default="fla-hub/gla-340M-15B")
    parser.add_argument("--layer_idx",  type=int,   default=3)
    parser.add_argument("--sigma",      type=float, default=0.3)
    parser.add_argument("--seq_len",    type=int,   default=32)
    parser.add_argument("--proj_dim",   type=int,   default=0,
                        help="Bottleneck 投影维度 (0=不投影；>0 时因果结构在低维空间闭环)")
    parser.add_argument("--n_train",    type=int,   default=100000)
    parser.add_argument("--n_val",      type=int,   default=10000)
    parser.add_argument("--n_test",     type=int,   default=10000)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--rescale_mode", type=str, default="none",
                        choices=["none", "rms_rescale", "partial_rescale",
                                 "input_only_rescale", "l2_normalize", "center_only"],
                        help="Bottleneck rescale 策略 (default: none)")
    parser.add_argument("--rescale_alpha", type=float, default=1.0,
                        help="partial_rescale 的插值系数 (default: 1.0, 等价于 rms_rescale)")
    parser.add_argument("--noise_type", type=str, default="process",
                        choices=["process", "observation"],
                        help="噪声类型: process=噪声参与动力学, observation=噪声只加在观测上")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"[config] output dir: {args.data_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TeacherConfig(model_name=args.model_name, layer_idx=args.layer_idx)
    teacher = ContinuousHSpaceTeacher(cfg).to(device)
    D_internal = teacher.d_hidden
    L   = args.seq_len
    sig = args.sigma

    W_gpu = None
    if args.proj_dim > 0 and args.proj_dim < D_internal:
        d = args.proj_dim
        W = make_ortho_proj(d, D_internal, seed=42)
        W_gpu = torch.from_numpy(W).to(device)
        D_out = d
        print(f"[config] projection: {D_internal} -> {d} (orthonormal)")
    else:
        D_out = D_internal
        print(f"[config] no projection, D_out={D_out}")

    rescale_mode = args.rescale_mode if W_gpu is not None else 'none'
    print(f"[config] D_internal={D_internal}, D_out={D_out}, L={L}, sigma={sig}")
    print(f"[config] rescale_mode={rescale_mode}, rescale_alpha={args.rescale_alpha}")
    print(f"[config] noise_type={args.noise_type}")
    print(f"[config] emb_scale={teacher.emb_scale:.4f}")
    print(f"[config] train={args.n_train}, val={args.n_val}, test={args.n_test}")

    splits = [
        ("train", args.n_train),
        ("val",   args.n_val),
        ("test",  args.n_test),
    ]

    for split, n in splits:
        vec_path  = os.path.join(args.data_dir, f"{split}_vectors.npy")
        init_path = os.path.join(args.data_dir, f"{split}_init_vectors.npy")
        vec_mmap  = np.memmap(vec_path,  dtype="float32", mode="w+", shape=(n, L, D_out))
        init_mmap = np.memmap(init_path, dtype="float32", mode="w+", shape=(n, 1, D_out))
        fill_split(teacher, n, L, sig, args.batch_size, device,
                   vec_mmap, init_mmap, split, W_gpu=W_gpu,
                   rescale_mode=rescale_mode,
                   rescale_alpha=args.rescale_alpha,
                   noise_type=args.noise_type)
        del vec_mmap, init_mmap
        print(f"[{split}] saved → {vec_path}")

    config = {
        "seq_length":    L,
        "vector_dim":    D_out,
        "d_internal":    D_internal,
        "num_init":      1,
        "train_samples": args.n_train,
        "val_samples":   args.n_val,
        "test_samples":  args.n_test,
        "sigma":         sig,
        "emb_scale":     teacher.emb_scale,
        "model_name":    args.model_name,
        "layer_idx":     args.layer_idx,
        "proj_dim":      args.proj_dim if args.proj_dim > 0 else None,
        "rescale_mode":  rescale_mode,
        "rescale_alpha": args.rescale_alpha,
        "noise_type":    args.noise_type,
    }
    if W_gpu is not None:
        config["proj_matrix"] = W_gpu.cpu()
    torch.save(config, os.path.join(args.data_dir, "data_config.pt"))
    print(f"[done] data_config.pt saved → {args.data_dir}")


if __name__ == "__main__":
    main()
