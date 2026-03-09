"""
FLA Teacher: 加载 fla-hub/gla-340M-15B 预训练权重，截取前 N 层提取 hidden state。

截取策略：加载全部权重后，只保留 model.model.layers[:layer_idx+1]，
丢弃后续层，节省 forward 计算。hook 挂在截取后的最后一层。

使用方式:
    teacher = FLATeacher(cfg)
    result  = teacher.extract(input_ids)   # input_ids: [B, L] LongTensor（CUDA）
    # result['hidden']:   [B, L, D]  原始因果顺序 hidden state
    # result['shuffled']: [B, L, D]  每条序列独立随机打乱
    # result['perm']:     [B, L]     打乱置换
    # result['order']:    [B, L]     arange(L)

特性:
  - 必须在 CUDA 上运行（fla Triton kernel 要求）
  - layer_idx=3 → 使用前 4 层；layer_idx=5 → 使用前 6 层
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FLATeacher(nn.Module):
    """
    从 fla-hub/gla-340M-15B（Gated Linear Attention，24层，hidden_size=1024）
    截取前 (layer_idx+1) 层，提取该层的 hidden state 作为训练数据。
    """

    def __init__(self, cfg):
        super().__init__()
        from fla.models import GLAForCausalLM

        self.cfg = cfg
        print(f"[FLATeacher] Loading {cfg.model_name} ...")
        model = GLAForCausalLM.from_pretrained(cfg.model_name)

        # 截取前 (layer_idx+1) 层，丢弃后续层节省计算
        n_keep = cfg.layer_idx + 1
        model.model.layers = nn.ModuleList(
            list(model.model.layers)[:n_keep]
        )
        print(f"[FLATeacher] Truncated to {n_keep} layers "
              f"(layer_idx={cfg.layer_idx}, d_model={model.config.hidden_size})")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model    = model
        self.d_hidden = model.config.hidden_size
        self._hidden_buf = None

        # hook 挂在截取后的最后一层
        def _hook(module, inp, output):
            # GLABlock → (hidden_states [B,L,D], attentions, cache)
            h = output[0] if isinstance(output, tuple) else output
            self._hidden_buf = h.detach()

        self.model.model.layers[-1].register_forward_hook(_hook)

    @torch.no_grad()
    def extract(self, input_ids: torch.Tensor) -> dict:
        """
        input_ids: [B, L] LongTensor（需在 CUDA 上）
        返回: hidden, order, shuffled, perm（均在同一 device）
        """
        B, L = input_ids.shape
        self._hidden_buf = None
        # forward 只经过前 N 层（后续层已被截断）
        self.model(input_ids)
        hidden = self._hidden_buf
        assert hidden is not None, "Hook 未触发"

        device    = hidden.device
        perms     = torch.stack([
            torch.randperm(L, device=device) for _ in range(B)
        ])
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        shuffled  = hidden[batch_idx, perms]
        order     = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        return {
            "hidden":   hidden,
            "order":    order,
            "shuffled": shuffled,
            "perm":     perms,
        }


class ContinuousHSpaceTeacher(nn.Module):
    """
    连续 h-space AR Teacher：用预训练 GLA 前 (layer_idx+1) 层在连续空间自回归生成序列。

    生成过程（v2: 使用 GLA 真实 embedding 范数作为输入尺度）：
      s = mean(||GLA.embed||)             (预训练 embedding 的平均范数，~1.32)
      h_0 = Normalize(N(0,I_D)) × s
      x_{t-1} = Normalize(h_{t-1}) × s   (归一化到 GLA 期望的输入尺度)
      μ_t = GLA_4L(inputs_embeds=[x_0,...,x_{t-1}])[:, -1, :]
      ε_t ~ N(0, σ² I_D)
      h_t = μ_t + ε_t                    (t = 1..L-1)

    注：必须在 CUDA 上运行（FLA Triton kernel 要求）。
    """

    def __init__(self, cfg):
        super().__init__()
        from fla.models import GLAForCausalLM

        self.cfg = cfg
        print(f"[ContinuousHSpaceTeacher] Loading {cfg.model_name} ...")
        model = GLAForCausalLM.from_pretrained(cfg.model_name)

        n_keep = cfg.layer_idx + 1
        model.model.layers = nn.ModuleList(list(model.model.layers)[:n_keep])
        print(f"[ContinuousHSpaceTeacher] Truncated to {n_keep} layers "
              f"(layer_idx={cfg.layer_idx}, d_model={model.config.hidden_size})")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model    = model
        self.d_hidden = model.config.hidden_size
        self._hidden_buf = None

        emb_w = model.model.embeddings.weight.data
        self.emb_scale = emb_w.norm(dim=-1).mean().item()
        print(f"[ContinuousHSpaceTeacher] embedding scale = {self.emb_scale:.4f} "
              f"(was sqrt(D)={math.sqrt(self.d_hidden):.1f})")

        def _hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self._hidden_buf = h.detach()

        self.model.model.layers[-1].register_forward_hook(_hook)

    @torch.no_grad()
    def generate_sequence(self, B: int, L: int, sigma: float,
                          device: torch.device,
                          W: torch.Tensor = None,
                          rescale_mode: str = 'none',
                          rescale_alpha: float = 1.0,
                          noise_type: str = 'process') -> torch.Tensor:
        """
        生成 B 条长度为 L 的序列。

        当 W=None 时，在 D 维空间生成（原始模式）：
          返回: [B, L, D]

        当 W 不为 None 时，在 d 维空间生成（bottleneck 模式）：
          W: [d, D] 正交投影矩阵
          返回: [B, L, d]

        rescale_mode (bottleneck 模式):
          'none'            — 不做 rescale（原始行为，会坍缩到 attractor）
          'rms_rescale'     — RMS rescale 到 emb_scale 对应的 per-dim RMS
          'partial_rescale' — 插值: (1-alpha)*mu_raw + alpha*mu_rms_scaled，
                              保留部分幅度演化，alpha 由 rescale_alpha 控制
          'input_only_rescale' — 只对输入做 RMS rescale，输出不做 rescale，
                                 保留真实状态动力学，同时避免输入尺度失控
          'l2_normalize'    — L2 归一化到 emb_scale（与 no_proj 模式一致）
          'center_only'     — 仅减去均值（诊断用，判断 DC bias 贡献）

        rescale_alpha: partial_rescale 模式的插值系数，alpha=1.0 等价于 rms_rescale

        noise_type:
          'process'     — h_{t+1} = f(h_{0:t}) + ε，噪声参与状态演化（当前默认）
          'observation' — h_{t+1} = f(h_{0:t})，teacher 内部干净闭环；
                          返回 y_t = h_t + ε 给 student，理论下界 = σ²·d
        """
        D = self.d_hidden
        s = self.emb_scale

        if W is not None:
            d = W.shape[0]
            # target_rms: 对齐 no_proj 模式的 per-dim RMS = emb_scale / sqrt(d)
            target_rms = s / math.sqrt(d)

            def _rescale(mu):
                """对 mu [B, d] 应用 rescale_mode 指定的尺度控制。"""
                if rescale_mode == 'none' or rescale_mode == 'input_only_rescale':
                    return mu
                elif rescale_mode == 'rms_rescale':
                    rms = mu.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
                    return mu * (target_rms / rms)
                elif rescale_mode == 'partial_rescale':
                    rms = mu.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
                    mu_scaled = mu * (target_rms / rms)
                    return (1 - rescale_alpha) * mu + rescale_alpha * mu_scaled
                elif rescale_mode == 'l2_normalize':
                    return F.normalize(mu, dim=-1) * s
                elif rescale_mode == 'center_only':
                    return mu - mu.mean(dim=-1, keepdim=True)
                else:
                    raise ValueError(f"Unknown rescale_mode: {rescale_mode}")

            def _input_transform(h):
                """将 d 维 h 变换为 D 维输入喂给 GLA。"""
                if rescale_mode == 'input_only_rescale':
                    # 对 h 做 RMS rescale 到 target_rms，再投影上去
                    rms = h.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
                    h_rescaled = h * (target_rms / rms)
                    return h_rescaled @ W
                else:
                    # 默认：L2 归一化到 emb_scale，再投影上去
                    return F.normalize(h, dim=-1) * s @ W

            # ── Bottleneck 模式：在 d 维空间生成 ──
            # h_0 初始 scale 根据 rescale_mode 确定
            if rescale_mode in ('l2_normalize', 'rms_rescale',
                                'partial_rescale', 'input_only_rescale'):
                # 直接用 emb_scale 对齐的尺度
                h0 = torch.randn(B, d, device=device)
                h0 = F.normalize(h0, dim=-1) * s
            else:
                # Warmup forward 估算输出 norm（原始行为）
                h0_dir = torch.randn(B, d, device=device)
                h0_dir = F.normalize(h0_dir, dim=-1) * s
                x0_up = h0_dir @ W
                self._hidden_buf = None
                self.model(inputs_embeds=x0_up.unsqueeze(1))
                warmup_out = self._hidden_buf[:, -1, :]
                h0_scale = (warmup_out @ W.T).norm(dim=-1).mean().item()

                h0 = torch.randn(B, d, device=device)
                h0 = F.normalize(h0, dim=-1) * h0_scale

            seq = [h0]  # latent states (clean for observation noise)

            for t in range(1, L):
                # d→D: 输入变换后投影上去喂 GLA
                x_hist = torch.stack(
                    [_input_transform(h) for h in seq],
                    dim=1,
                )  # [B, t, D]

                self._hidden_buf = None
                self.model(inputs_embeds=x_hist)

                assert self._hidden_buf is not None, "Hook 未触发"
                out_t = self._hidden_buf[:, -1, :]   # [B, D]

                # 1024→d: 投影下来 + rescale
                mu_t = out_t @ W.T                    # [B, d]
                mu_t = _rescale(mu_t)

                if noise_type == 'observation':
                    # 干净状态闭环，噪声不参与动力学
                    seq.append(mu_t)
                else:
                    # process noise：噪声参与下一步动力学
                    h_t = mu_t + torch.randn_like(mu_t) * sigma
                    seq.append(h_t)

            latent = torch.stack(seq, dim=1)  # [B, L, d]

            if noise_type == 'observation':
                # 对整个序列加观测噪声：y_t = h_t + ε
                return latent + torch.randn_like(latent) * sigma
            else:
                return latent

        # ── 原始模式：在 D 维空间生成 ──
        h0 = torch.randn(B, D, device=device)
        h0 = F.normalize(h0, dim=-1) * s

        seq = [h0]

        for t in range(1, L):
            x_hist = torch.stack(
                [F.normalize(h, dim=-1) * s for h in seq],
                dim=1,
            )  # [B, t, D]

            self._hidden_buf = None
            self.model(inputs_embeds=x_hist)

            assert self._hidden_buf is not None, "Hook 未触发"
            mu_t = self._hidden_buf[:, -1, :]  # [B, D]
            mu_t = F.normalize(mu_t, dim=-1) * s  # 归一化到统一尺度

            if noise_type == 'observation':
                seq.append(mu_t)
            else:
                h_t = mu_t + torch.randn_like(mu_t) * sigma
                seq.append(h_t)

        latent = torch.stack(seq, dim=1)  # [B, L, D]
        if noise_type == 'observation':
            return latent + torch.randn_like(latent) * sigma
        else:
            return latent
