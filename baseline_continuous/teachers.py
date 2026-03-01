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
import torch
import torch.nn as nn


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
