"""
GPT-2 Teacher: 用冻结的 GPT-2 前 4 层自回归生成连续向量序列。

生成公式:
    x_0   ~ Normalize(N(0,I)) × √D
    x_t   = Normalize( GPT2_4L([x_0...x_{t-1}])[-1] + ε ) × √D
    ε     ~ N(0, noise_scale × I)

关键: 缩放因子 √D 防止 attention temperature collapse。
"""
import math
import torch
import torch.nn.functional as F
from transformers import GPT2Model


class GPT2Teacher(torch.nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        gpt2 = GPT2Model.from_pretrained('gpt2')
        self.h    = torch.nn.ModuleList(gpt2.h[:num_layers])
        self.ln_f = gpt2.ln_f
        for p in self.parameters():
            p.requires_grad_(False)
        self.D     = gpt2.config.n_embd    # 768
        self.scale = math.sqrt(self.D)

    @torch.no_grad()
    def forward(self, x):
        """x: [B, T, D] → [B, T, D]"""
        for block in self.h:
            x = block(x)[0]
        return self.ln_f(x)

    @torch.no_grad()
    def generate_sequence(self, length, batch_size, noise_scale=5.0,
                          init_mode='positive_first', device=None):
        """
        生成 [batch_size, length, D] 的序列。
        init_mode: 'positive_first' → x_0[0]>0; 'negative_first' → x_0[0]<0
        返回 dict: vectors [B,L,D], init_vectors [B,1,D]
        """
        if device is None:
            device = next(self.parameters()).device
        D, scale = self.D, self.scale
        seqs = torch.zeros(batch_size, length, D, device=device)

        # x_0: 归一化随机向量 × √D，满足 init_mode 约束
        x0 = F.normalize(torch.randn(batch_size, D, device=device), dim=-1) * scale
        if init_mode == 'positive_first':
            x0[:, 0] = x0[:, 0].abs()
        else:
            x0[:, 0] = -x0[:, 0].abs()
        seqs[:, 0] = x0

        for t in range(1, length):
            out   = self.forward(seqs[:, :t])[:, -1, :]        # [B, D]
            noise = torch.randn_like(out) * noise_scale
            seqs[:, t] = F.normalize(out + noise, dim=-1) * scale

        num_init = 1
        return {
            'vectors':      seqs,
            'init_vectors': seqs[:, :num_init],
        }
