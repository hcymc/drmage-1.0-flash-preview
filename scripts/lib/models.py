"""models.py — 生成模型（64×64 RGBA / 4 通道）。

三个可选方案，按「先跑通再升级」的顺序实现：

1. :class:`DCGANGenerator` / :class:`DCGANDiscriminator`
   最轻的基线。4 层反卷积，参数量 ~3.5M。几十秒一个 epoch，用来快速验证
   数据管线和 alpha 通道的处理方式是否正确。

2. :class:`SmallUNet` + :class:`DDPM`
   轻量扩散。base_channels 默认 48、3 个下采样层，在 64×64 上参数量 ~10M，
   AMP 下 batch 32 大约吃 2-3GB 显存。

3. VQ-VAE 在 ``lib/vqvae.py`` 单独实现（编码器/解码器 + 码本 + 自回归先验）。

**alpha 处理**：所有生成器都输出 4 通道。RGB 走 ``tanh``，alpha 走 ``sigmoid``。
理由见 ``README`` 的「alpha 通道设计」一节——把 alpha 和 RGB 拆成两个独立头，
是为了让 alpha 的梯度不被 RGB 的重建/对抗梯度带偏；墨迹式的半透明边缘在
皮肤里很少见，alpha 更接近二值，用 sigmoid + 二值化正则更贴。
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 公共积木
# --------------------------------------------------------------------------

class SpectralNorm(nn.Module):
    """轻量封装：优先用 PyTorch 内置 ``spectral_norm``。"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = nn.utils.spectral_norm(module)

    def forward(self, x):
        return self.module(x)


def apply_sn(conv: nn.Module, use_sn: bool) -> nn.Module:
    """按需给卷积加谱归一化。

    **``use_sn=False`` 不是可选项，而是启用 R1 时必须做的事**：
    谱归一化的权重是每次 forward 用幂迭代算出来的，而 R1 需要对 D 的输出做
    **二阶求导**（``create_graph=True``）。二阶图会穿过幂迭代的反馈回路，
    把 ``‖∇D‖`` 放大到数值崩坏 —— 实测 ``r1`` 从 0.002 一路雪崩到 **8056**，
    整个判别器被毁掉。StyleGAN 用 R1 时本来就不加谱归一化（R1 是它的替代品）。
    """
    return nn.utils.spectral_norm(conv) if use_sn else conv


class SelfAttention(nn.Module):
    """单头自注意力（判别器与 UNet 瓶颈共用）。"""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x)
        q, k, v = self.qkv(y).chunk(3, dim=1)
        # 三个张量都要把「空间位置」搬到序列维（-2）、「每头通道」留在特征维（-1）：
        #   (b, c, h, w) -> (b, heads, c/heads, hw) -> (b, heads, hw, c/heads)
        # 早期版本只给 q / v 做了 transpose，k 漏了 —— 于是 SDPA 收到
        # query (L=hw, E=c/H) 与 key (S=c/H, E=hw)，语义完全错位。
        # 它之所以一直没报错，是因为 DCGAN 判别器 base=64 时 c/H = 64 恰好等于
        # hw = 64，两个维度撞上了；一旦把 base 提到 128（c/H=128）就立刻抛
        # "Expected size for first two dimensions of batch2 tensor to be:
        #  [1024, 128] but got: [1024, 64]"。
        q = q.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)
        k = k.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)
        v = v.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


# --------------------------------------------------------------------------
# 方案 1：DCGAN
# --------------------------------------------------------------------------

def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FiLM(nn.Module):
    """特征级条件注入：``h ← (1+γ)⊙h + β``，``(γ, β) = MLP(c)``。

    为什么不用「把条件拼到 z 上」：实测那是最弱的注入方式 —— 判别器很容易
    **直接忽略条件**（只判局部纹理就够拿分），于是生成器也没有动力去用条件，
    最后条件形同虚设。FiLM 把条件乘性/加性地写进**每一层特征**，
    通路短、梯度直接。

    **MLP 最后一层零初始化**是关键：γ=β=0 时 ``(1+0)⊙h+0 = h``，
    训练起点严格等价于无条件模型，不会一上来就被随机条件扰动。

    参考：条件向量的离散段见 ``labelset.describe``（19..35 是 one-hot / 布尔位）。
    """

    def __init__(self, cond_dim: int, ch: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(64, cond_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, ch * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.net(cond)
        g, b = gb.chunk(2, dim=1)
        return h * (1.0 + g[:, :, None, None]) + b[:, :, None, None]


#: 条件向量里「色相直方图」占据的下标区间（见 labelset.describe），argmax 即色相桶
HUE_HIST_SLICE = slice(0, 12)
#: 色相桶数（12 个 30° 桶）
HUE_BINS = 12


def hue_bin_of(cond: torch.Tensor) -> torch.Tensor:
    """从条件向量取离散色相桶下标 ``(B,)``，给调色板调制用。"""
    return cond[:, HUE_HIST_SLICE].argmax(dim=1)


class PaletteHead(nn.Module):
    """把 RGB 头从「直接回归 3 通道」换成「K 个调色板色的**硬选择**」。

    数学形式（straight-through）::

        logits = Conv(h) / τ                     ∈ R^{B×K×H×W}
        p_soft = softmax_k(logits)
        p_hard = onehot(argmax_k p_soft)
        p      = p_hard + p_soft − p_soft.detach()    # 前向用硬、反向走软

        I      = Σ_k p_k · P_k = einsum('bkhw,kc->bchw', p, P)

    **为什么必须用硬选择（STE）**：一开始我写成纯软凸组合
    ``I = Σ softmax(logits)·P``，以为「图像最多 K 种颜色」——
    实测**不成立**：软混合的结果落在调色板凸包**内部**，每个像素混合比不同，
    颜色数照样能到 4000+。要真正把颜色数封顶，前向必须走 one-hot。
    梯度用 straight-through 从软分布回传（这是 VQ-VAE 的标准做法）。

    ``P ∈ R^{K×3}`` 可学习，用真实可见像素的 k-means 初始化
    （``models/palette_k96.npy``，实测真实唯一色数中位 73，K=96 足够覆盖）。

    **色相调制**（``hue_bins > 0``）—— 对「颜色可控」最直接的一条通路::

        P' = P ⊙ (1 + W[hue_bin]) + b[hue_bin]

    只有 ``2 × 12 × 3 = 72`` 个参数，却把「指定色相 → 整体色调跟着变」变成
    架构上的直接通路，而不是指望网络自己从条件向量里学出来。
    两个 Embedding 都零初始化 ⇒ 起点是无调制的纯调色板。

    权威判据是 ``_probe/loss_smoke.py`` 里的检查：输出唯一色数必须 ≤ K。
    """

    def __init__(self, in_ch: int, k: int = 96,
                 init_palette: "torch.Tensor | None" = None,
                 hue_bins: int = 0, tau_init: float = 0.5):
        super().__init__()
        self.k = k
        self.hue_bins = hue_bins
        self.tau_init = tau_init
        self.conv = nn.Conv2d(in_ch, k, 3, 1, 1)
        if init_palette is not None:
            assert init_palette.shape == (k, 3)
            self.palette = nn.Parameter(init_palette.clone().float())
        else:
            g = torch.linspace(0, 1, k)
            self.palette = nn.Parameter(
                torch.stack([g, g.flip(0), torch.ones_like(g)], dim=1))
        if hue_bins > 0:
            self.hue_scale = nn.Embedding(hue_bins, 3)
            self.hue_bias = nn.Embedding(hue_bins, 3)
            nn.init.zeros_(self.hue_scale.weight)
            nn.init.zeros_(self.hue_bias.weight)

    def effective_palette(self, hue_bin: "torch.Tensor | None" = None) -> torch.Tensor:
        """按色相桶调制后的调色板 ``(K,3)``（同一 batch 内共用一个桶）。"""
        if hue_bin is None or self.hue_bins <= 0:
            return self.palette
        idx = hue_bin.reshape(-1)[0]                       # 取 batch 首个，保持 (K,3)
        return self.palette * (1.0 + self.hue_scale(idx)) + self.hue_bias(idx)

    def forward(self, h: torch.Tensor, tau: float = 0.5,
                hue_bin: "torch.Tensor | None" = None) -> torch.Tensor:
        # **整个头必须在 fp32 下跑**：STE 里要做 ``p_hard + p_soft − p_soft.detach()``，
        # fp16 下 argmax 附近的两个大数相减会丢有效位，量化的「硬」会退化。
        # 这是个 3×3 的小卷积，关掉 autocast 的代价可以忽略。
        dt = h.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            hf = h.float()
            pal = self.effective_palette(hue_bin).float()
            p_soft = torch.softmax(self.conv(hf) / tau, dim=1)
            # 留下软分配供训练循环算「使用熵护栏」（见 losses.palette_usage_guard）。
            # 必须保留计算图（不要 detach），否则护栏拿不到梯度。
            self.last_probs = p_soft
            idx = p_soft.argmax(dim=1, keepdim=True)
            p_hard = torch.zeros_like(p_soft).scatter_(1, idx, 1.0)
            p = p_hard + p_soft - p_soft.detach()
            out = torch.einsum("bkhw,kc->bchw", p, pal)
        return out.to(dt)


class DCGANGenerator(nn.Module):
    """z(128) → 4×4 → 8 → 16 → 32 → 64。

    ``palette_k > 0`` 时 RGB 走 :class:`PaletteHead`（结构性限制颜色数），
    alpha 仍走 sigmoid + 二值化 hinge（实测这一路已经收敛得很好）。
    ``film=True`` 时用 :class:`FiLM` 在每一层特征上注入条件。
    """

    def __init__(self, z_dim: int = 128, base: int = 256, cond_dim: int = 0,
                 palette_k: int = 0, init_palette=None, film: bool = False,
                 palette_hue_bins: int = 0, tau_init: float = 0.5,
                 alpha_mode: str = "head"):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.palette_k = palette_k
        self.film = film and cond_dim > 0
        self.alpha_mode = alpha_mode
        self.fc = nn.Linear(z_dim + cond_dim, base * 4 * 4, bias=False)
        self.bn0 = nn.BatchNorm2d(base)
        self.up1 = _up_block(base, base // 2)        # 8
        self.up2 = _up_block(base // 2, base // 4)   # 16
        self.up3 = _up_block(base // 4, base // 8)   # 32
        self.up4 = _up_block(base // 8, base // 16)  # 64
        if self.film:
            # 与 up1..up4 的输出通道一一对应
            self.films = nn.ModuleList([
                FiLM(cond_dim, base),
                FiLM(cond_dim, base // 2),
                FiLM(cond_dim, base // 4),
                FiLM(cond_dim, base // 8),
                FiLM(cond_dim, base // 16),
            ])
        if palette_k > 0:
            self.head_rgb = PaletteHead(base // 16, palette_k, init_palette,
                                        hue_bins=palette_hue_bins, tau_init=tau_init)
        else:
            self.head_rgb = nn.Sequential(
                nn.Conv2d(base // 16, base // 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(base // 32), nn.ReLU(inplace=True),
                nn.Conv2d(base // 32, 3, 3, 1, 1))
        # alpha_mode="none"：alpha 由 UV 模板在训练/生成侧合成，**模型完全不学**。
        # 三个 MC 皮肤生成项目（SDXL+LoRA / SD1.5+LoRA / 像素扩散）全这么做——
        # 实测 base 层 98~100% 恒满、overlay 是两级伯努利、padding 恒空，
        # alpha 在这个域里几乎是确定性函数，用 sigmoid 头学它等于把容量浪费在掷硬币上，
        # 而且这正是「剪影是噪点块」的直接原因之一。
        # "none" 与 "template" 都不建 alpha 头（template 的 alpha 由 UV 模板合成）
        self.head_a = None if alpha_mode in ("none", "template") else nn.Sequential(
            nn.Conv2d(base // 16, base // 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base // 32), nn.ReLU(inplace=True),
            nn.Conv2d(base // 32, 1, 3, 1, 1))

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None,
                tau: float = 0.5):
        if self.film and cond is None:
            raise ValueError("film=True 时 forward 必须提供 cond")
        if self.cond_dim and cond is not None:
            z = torch.cat([z, cond], dim=1)
        # 注意顺序：BatchNorm2d 要求 4D 输入，所以必须**先 reshape 再 bn**。
        # 早期写法是 bn0(fc(z)).reshape(...)，bn0 拿到的是 (B, C) 的 2D 张量，
        # 直接抛 "expected 4D input (got 2D input)"。
        h = self.bn0(self.fc(z).reshape(z.size(0), -1, 4, 4))
        if self.film:
            h = self.films[0](h, cond)
        for i, blk in enumerate((self.up1, self.up2, self.up3, self.up4), start=1):
            h = blk(h)
            if self.film:
                h = self.films[i](h, cond)
        hue_bin = hue_bin_of(cond) if (self.palette_k > 0 and cond is not None) else None
        if self.palette_k > 0:
            rgb = self.head_rgb(h, tau, hue_bin)          # 已在 [0,1]
        else:
            rgb = torch.tanh(self.head_rgb(h)) * 0.5 + 0.5
        if self.alpha_mode in ("none", "template"):
            return rgb                                    # (B,3,64,64)
        a = torch.sigmoid(self.head_a(h))
        return torch.cat([rgb, a], dim=1)                # 统一到 [0,1]


class DCGANDiscriminator(nn.Module):
    """带自注意力的判别器，输出 ``grid×grid`` 的 patch 真伪 logits。

    ``grid`` 是输出网格边长，patch 像素大小 = 64/grid：
    grid=8 → 8×8 像素 patch（旧行为）；grid=32 → **2×2 像素 patch**。

    **为什么 2×2 更好**：像素画 GAN 的实证结论（arXiv:2208.06413）——
    低分辨率下每个像素信息量极大、低频区域极小，2×2 patch 的判别器
    **同时判纹理和形状边缘**，显著优于大 patch（他们试了 2/5/11/64，
    2×2 最接近真值；单 patch 模型还会在轮廓外产出「悬空像素」）。

    ``features(x, cond)`` 返回池化后的特征向量，供**特征匹配损失**使用
    （见 20_train_dcgan.py：要求 G 的输出在 D 的特征空间里与真实图无法区分）。

    ``use_sn=True`` 时加谱归一化；**启用 R1 时必须传 False**（二阶求导
    会穿过幂迭代导致数值崩坏，见 :func:`apply_sn`）。
    """

    def __init__(self, base: int = 64, cond_dim: int = 0, use_sn: bool = True,
                 grid: int = 8):
        super().__init__()
        assert grid in (8, 16, 32), f"grid 只支持 8/16/32，收到 {grid}"
        self.cond_dim = cond_dim
        self.use_sn = use_sn
        self.grid = grid
        n_down = int(round(np.log2(64 // grid)))
        layers: list[nn.Module] = []
        cin = 4 + cond_dim
        for i in range(n_down):
            cout = base * (2 ** min(i, 2))
            layers.append(apply_sn(nn.Conv2d(cin, cout, 4, 2, 1, bias=(i == 0)), use_sn))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            cin = cout
        # 网格越大、下采样越少，补 stride-1 卷积维持容量（否则 grid=32 时 D 只有
        # ~0.09M 参数、一个下采样层，感受野太小撑不起特征匹配）
        for _ in range(2 if grid >= 32 else (1 if grid == 16 else 0)):
            layers.append(apply_sn(nn.Conv2d(cin, cin, 3, 1, 1, bias=False), use_sn))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
        self.ch = cin
        # 注意力：32×32 网格 = 1024 个 token，batch 192 时注意力矩阵要 ~3GB，不划算；
        # 16×16 及以下才加
        self.attn = SelfAttention(self.ch) if grid <= 16 else nn.Identity()
        # 1×1 的 out 头**不能带 padding**：padding=1 会把 32×32 垫成 34×34，
        # 与投影项的 32×32 形状对不上（实测一 forward 就抛 shape mismatch）
        k = 3 if grid == 8 else 1
        self.out = apply_sn(nn.Conv2d(self.ch, 1, k, 1, 1 if k == 3 else 0), use_sn)

    def features(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """骨干网输出在空间上取均值 → ``(B, C)`` 特征向量（特征匹配用）。"""
        h = self.backbone(x, cond)
        return h.mean(dim=(2, 3))

    def backbone(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.cond_dim and cond is not None:
            c = cond.reshape(cond.size(0), self.cond_dim, 1, 1).expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
        return self.attn(self.net(x))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        return self.out(self.backbone(x, cond))


class ProjectionDiscriminator(nn.Module):
    """Projection 判别器（Miyato & Koyama, 2018）::

        D(x, c) = out(f(x)) + <v(c), f(x)>

    ``f(x)`` 是卷积塔输出的特征图 ``(B, C, H', W')``，无条件项走 1×1 卷积成
    patch logits，条件项把 ``f`` 在通道上求和得到向量 ``(B, C)`` 再与 ``v(c)``
    做内积、广播回空间维。

    **为什么换掉「条件当额外通道平面」**：那种写法把条件与图像在**输入层**拼接，
    判别器要自己学会「把某个平面上的常数和图像联系起来」，通路极长，
    实测结果就是**判别器直接无视条件**（数据集里 99.8% 是 classic，
    不看条件也能拿到很高准确率）。Projection 把条件放在**输出侧**做内积，
    条件与「这张图的特征」的关系是一步乘法，判别器想忽略都难。
    """

    def __init__(self, base: int = 64, cond_dim: int = 0, use_sn: bool = True,
                 grid: int = 8):
        super().__init__()
        assert grid in (8, 16, 32), f"grid 只支持 8/16/32，收到 {grid}"
        self.cond_dim = cond_dim
        self.use_sn = use_sn
        self.grid = grid
        n_down = int(round(np.log2(64 // grid)))
        layers: list[nn.Module] = []
        cin = 4
        for i in range(n_down):
            cout = base * (2 ** min(i, 2))
            layers.append(apply_sn(nn.Conv2d(cin, cout, 4, 2, 1, bias=(i == 0)), use_sn))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            cin = cout
        if grid >= 16:
            layers.append(apply_sn(nn.Conv2d(cin, cin, 3, 1, 1, bias=False), use_sn))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
        self.ch = cin
        self.attn = SelfAttention(self.ch) if grid <= 16 else nn.Identity()
        k = 3 if grid == 8 else 1
        self.out = apply_sn(nn.Conv2d(self.ch, 1, k, 1, 1 if k == 3 else 0), use_sn)
        if cond_dim:
            # 零初始化 ⇒ 起点等价于无条件 patch 判别器（同样的理由：不要一上来就扰动）
            self.proj = nn.Linear(cond_dim, self.ch)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def backbone(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.attn(self.net(x))

    def features(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.backbone(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        f = self.backbone(x)
        y = self.out(f)
        if self.cond_dim and cond is not None:
            v = self.proj(cond)                          # (B, C)
            s = torch.einsum("bc,bchw->bhw", v, f)       # (B, H', W')
            y = y + s[:, None]
        return y


# --------------------------------------------------------------------------
# 方案 2：DDPM + 小型 UNet
# --------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """扩散残差块：``t_emb`` 加性注入 +（可选）``cond`` 的 FiLM 乘性注入。

    **为什么条件不能再只加到 ``t_emb`` 上**：实测条件对输出的解释力很弱
    （overlay 盒 cond→均色 R² 只有 0.36，base 0.66），换条件时 16 张骨架
    是「同一个模子」。原因是 ``cond_proj(cond)`` 只和 ``t_emb`` 相加，
    再经每个块的 ``t_proj`` 线性层 —— 条件信号与时间信号共用同一条通路，
    会被时间信号淹没，且只有加性、没有乘性门控。

    FiLM 把条件写进**每一层特征**：``h ← (1+γ)⊙h + β``，通路短、梯度直接。
    ``γ=β=0`` 时是恒等映射，所以最后一层**零初始化**，训练起点严格等价于
    无条件模型，不会一上来就被随机条件带偏（StyleGAN 的 trick）。
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8,
                 cond_dim: int = 0):
        super().__init__()
        g1 = math.gcd(groups, in_ch) or 1
        g2 = math.gcd(groups, out_ch) or 1
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.film = None
        if cond_dim > 0:
            self.film = nn.Sequential(
                nn.Linear(cond_dim, t_dim), nn.SiLU(),
                nn.Linear(t_dim, out_ch * 2),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    def forward(self, x, t_emb, cond=None):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        if self.film is not None and cond is not None:
            g, b = self.film(cond).chunk(2, dim=1)
            h = h * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """3 层下采样的小 UNet：64→32→16→8。

    base=48 时参数量约 10M，AMP 下 batch 32 在 8GB 卡上占用约 2-3GB。

    **输入通道 ≠ 输出通道**（``in_ch`` / ``out_ch`` 分开）：
    扩散张量只有 ``out_ch`` 个通道（RGB=3），但 UNet 的输入可以多带
    **条件平面** —— 典型用法是把 UV 模板 alpha 作为第 4 个输入通道拼在
    ``extra`` 里，见 :meth:`forward` 的 ``extra`` 参数。
    这样模型能看到「这个位置推理时会不会被露出来」，而不是只靠条件向量里
    两个标量（``overlay_used`` / ``transparent_ratio``）去猜。
    """

    def __init__(self, in_ch: int = 4, base: int = 48, t_dim: int = 256,
                 cond_dim: int = 0, attn_at: tuple[int, ...] = (0, 1),
                 n_mid: int = 2, out_ch: int | None = None):
        super().__init__()
        self.in_ch = in_ch          # UNet 输入通道数（含 extra 条件平面）
        # 扩散张量本身的通道数；缺省与 in_ch 相同（向后兼容旧的 4 通道 RGBA 模型）
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.t_dim = t_dim
        self.cond_dim = cond_dim
        if cond_dim:
            self.cond_proj = nn.Linear(cond_dim, t_dim)
        self.time_mlp = nn.Sequential(nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
                                      nn.Linear(t_dim * 2, t_dim))
        self.in_conv = nn.Conv2d(in_ch, base, 3, 1, 1)

        chs = [base, base * 2, base * 4]
        self.down1 = ResBlock(base, chs[0], t_dim, cond_dim=cond_dim)
        self.down2 = ResBlock(chs[0], chs[1], t_dim, cond_dim=cond_dim)
        self.down3 = ResBlock(chs[1], chs[2], t_dim, cond_dim=cond_dim)
        # 最深层（8×8）的通道数就是 chs[-1]（= down3 的输出）。
        # ⚠️ 早期这里写成 ``ResBlock(chs[i], chs[i])``，i=0 时按 base 建层，
        # 实际却收到 4*base 的张量，一 forward 就
        # "Expected weight to be a vector of size equal to the number of channels"
        # —— 这是第 5 个「从没真正跑通过」的 bug。
        self.downs = nn.ModuleList()
        for i in range(n_mid):
            self.downs.append(nn.ModuleList([
                ResBlock(chs[-1], chs[-1], t_dim, cond_dim=cond_dim),
                SelfAttention(chs[-1]) if i in attn_at else nn.Identity(),
            ]))
        self.mid1 = ResBlock(chs[-1], chs[-1], t_dim, cond_dim=cond_dim)
        self.mid_attn = SelfAttention(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], t_dim, cond_dim=cond_dim)
        # 解码器通道数必须按 **cat(上采样后的 h, skip)** 来算，不能想当然写 chs[i]*2：
        #   k=0: h=chs[2]@8↑, skip=chs[2]@16  -> 2*chs[2] -> chs[2]
        #   k=1: h=chs[2]@16↑, skip=chs[1]@32 -> chs[2]+chs[1] -> chs[1]
        #   k=2: h=chs[1]@32↑, skip=chs[0]@64 -> chs[1]+chs[0] -> chs[0]
        up_specs = [(chs[2] * 2, chs[2]),
                    (chs[2] + chs[1], chs[1]),
                    (chs[1] + chs[0], chs[0])]
        self.ups = nn.ModuleList()
        # ⚠️ 循环变量**绝不能叫 in_ch**：那会把 __init__ 的参数 in_ch 覆盖掉，
        # 于是下面 `nn.Conv2d(base, in_ch, ...)` 的输出通道变成
        # ``up_specs[-1][0] = chs[1]+chs[0] = 3*base``（base=48 时 144、
        # base=96 时 288），模型前向直接输出 144 通道的「图」。
        # 这个 bug 从最初写下来就在，扩散从来没跑通过。
        for up_in, up_out in up_specs:
            self.ups.append(nn.ModuleList([
                ResBlock(up_in, up_out, t_dim, cond_dim=cond_dim),
                ResBlock(up_out, up_out, t_dim, cond_dim=cond_dim),
            ]))
        self.out_norm = nn.GroupNorm(math.gcd(8, base), base)
        self.out_conv = nn.Conv2d(base, self.out_ch, 3, 1, 1)
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, cond=None, extra=None):
        """``extra``：与 ``x`` 在通道维拼接的**条件平面**（例如 UV 模板 alpha）。

        它和 ``x`` 走同一条输入路径但不参与扩散 —— 采样时它是**已知且恒定**的，
        所以每一轮去噪都重新拼一次。让模型看到「哪些像素会被露出来」，
        是把「第二层该画什么」从猜测变成**适定任务**的关键一步。
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.t_dim))
        if self.cond_dim and cond is not None:
            t_emb = t_emb + self.cond_proj(cond)
        if extra is not None:
            x = torch.cat([x, extra], dim=1)
        h = self.in_conv(x)
        skips = []
        # 每个 ResBlock 都必须拿到 t_emb（早期版本全程漏传，一跑就
        # "ResBlock.forward() missing 1 required positional argument: 't_emb'"）
        h = self.down1(h, t_emb, cond); skips.append(h); h = self.downsample(h)
        h = self.down2(h, t_emb, cond); skips.append(h); h = self.downsample(h)
        h = self.down3(h, t_emb, cond); skips.append(h); h = self.downsample(h)
        for res, attn in self.downs:
            h = attn(res(h, t_emb, cond))
        h = self.mid2(self.mid_attn(self.mid1(h, t_emb, cond)), t_emb, cond)
        for res1, res2 in self.ups:
            s = skips.pop()
            h = self.upsample(h)
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = res1(torch.cat([h, s], dim=1), t_emb, cond)
            h = res2(h, t_emb, cond)
        return self.out_conv(F.silu(self.out_norm(h)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """cosine schedule（Nichol & Dhariwal 2021）。

    为什么不用线性 beta：线性 schedule 在 64×64 这种小图上会把**大部分时间步**
    花在「几乎全噪」的状态，低噪段（真正决定细节的那些步）占比过小，
    小数据量下表现为「能出轮廓、细节糊」。cosine 把噪声预算平摊得更均匀。
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps / timesteps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    acp = f / f[0]
    betas = 1.0 - acp[1:] / acp[:-1]
    return betas.clamp(1e-8, 0.999).float()


class DDPM(nn.Module):
    """标准 DDPM（predict-epsilon）。默认 cosine schedule。"""

    def __init__(self, unet: SmallUNet, timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 schedule: str = "cosine"):
        super().__init__()
        self.unet = unet
        self.T = timesteps
        self.schedule = schedule
        # 扩散张量本身的通道数 = UNet 的**输出**通道数（in_ch 可能因 extra 条件平面更大）
        self.in_ch = getattr(unet, "out_ch", getattr(unet, "in_ch", 4))
        betas = (cosine_beta_schedule(timesteps) if schedule == "cosine"
                 else torch.linspace(beta_start, beta_end, timesteps))
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_acp", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_1macp", torch.sqrt(1.0 - self.alphas_cumprod))

    def q_sample(self, x0, t, noise):
        s1 = self.sqrt_acp[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_1macp[t].view(-1, 1, 1, 1)
        return s1 * x0 + s2 * noise

    def forward_step(self, x0, cond=None, extra=None, cond_dropout: float = 0.0):
        """返回 ``(pred_noise, noise, t, xt)``。

        拆出来是因为「模型自己学 alpha」的模式需要从 ``pred`` 反解 ``x0_hat``
        才能对 alpha 通道加二值化 hinge —— 只返回一个 MSE 标量就拿不到 ``pred``。

        ``cond_dropout``：以该概率把整条样本的条件置零。这是 **CFG 的训练前提**
        —— 不训 dropout 就直接在采样时做 ``eps_u + w(eps_c − eps_u)``，
        模型从没见过「无条件」这一侧，外推出来的方向是错的。
        """
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        cond_in = drop_cond(cond, cond_dropout)
        pred = self.unet(xt, t, cond_in, extra)
        return pred, noise, t, xt

    def loss(self, x0, cond=None, extra=None, cond_dropout: float = 0.0):
        pred, noise, _, _ = self.forward_step(x0, cond, extra, cond_dropout)
        return F.mse_loss(pred, noise)

    def masked_loss(self, x0, mask, cond=None, extra=None, weight=None,
                    cond_dropout: float = 0.0):
        """只在**可见像素**上算去噪损失（``mask`` 取 0/1，形状同 ``x0`` 的前两维）。

        为什么必须掩码
        --------------
        训练数据里 alpha==0 的像素 RGB 基本都是**纯黑**，而 overlay 盒有
        **80.5%** 的像素是透明的、padding 区 95.8% 透明。不掩码的话：

        * **>50% 的损失花在永远不会显示的像素上**（纯浪费）；
        * 更糟的是，overlay 位置的目标变成「80% 的样本里是纯黑」→
          条件均值 ≈ 黑 → 一旦 alpha 模板把 overlay 露出来，看到的就是一片黑。
          实测：16 张导出皮肤里 12 张的 overlay 近黑率 >43%，最大 100%。

        掩码后，overlay 位置只在「真实数据确实画了」的那 ~19.5% 样本上拿梯度，
        目标恢复成「真实皮肤在那里画什么」。

        ⚠️ 分母用**可见像素数**而不是总像素数，所以这一个数**不能和历史 mse 直接比**
        （历史值是全图像素均值，里面混着大量"预测黑色"的容易项）。

        ``weight``（可选，形状同 ``mask``）给不同区域不同权重。**这是治第二层的
        直接手段**：overlay 盒的有效可见像素只有 base 的 1/6.7（0.149 vs 0.994），
        掩码之后它的梯度像素比 base 少一个量级 —— 给 overlay 乘 3~6 才能把
        两个区域拉回同一个有效监督量级。权重进**分母**，所以它只改「哪个区域
        更重要」，不改整体损失的尺度。
        """
        pred, noise, _, _ = self.forward_step(x0, cond, extra, cond_dropout)
        se = (pred - noise) ** 2
        m = mask.expand_as(se)
        if weight is not None:
            m = m * weight.expand_as(se)
        return (se * m).sum() / m.sum().clamp(min=1.0)

    def predict_x0(self, xt, t, pred):
        """从预测噪声反解 x0（用于对生成结果加约束或做诊断）。"""
        acp = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return ((xt - torch.sqrt(1.0 - acp) * pred) / torch.sqrt(acp)).clamp(-1, 1)

    @torch.no_grad()
    def sample(self, n: int, shape=None, device="cuda",
               cond=None, ddim_steps: int = 100, eta: float = 0.0,
               extra=None, cfg_scale: float = 1.0,
               x_T: "torch.Tensor | None" = None, amp: bool = False):
        """DDIM 加速采样（100 步足够出可看的样本）。

        ``shape`` 缺省从 UNet 的 ``out_ch`` 推——**必须这样**：早期默认写死
        ``(4, 64, 64)``，一旦换成 3 通道（alpha 模板化）的模型，
        第一次前向就因为输入通道数不匹配而崩。

        ``extra``：条件平面（如 UV 模板 alpha），每步去噪都重新拼一次。
        它不参与扩散（不加噪），是**已知且恒定**的。

        ``cfg_scale``：classifier-free guidance 强度，``1.0`` = 关闭。

            eps = eps_uncond + cfg_scale · (eps_cond − eps_uncond)

        无条件一侧用**全零条件向量**，与训练时 ``cond_dropout`` 置零的路径
        完全一致（不能传 ``None`` —— 那会跳过 ``cond_proj`` 的 bias 和 FiLM，
        与训练侧见到的分布不是同一个）。

        ``x_T``：显式给初始噪声 ``(n, C, 64, 64)``。抽奖模式靠它保证
        **批量大小不影响结果**——初始噪声按候选序号一次性预生成，
        分几批送进来都是同一条噪声，结果与分批方式无关。

        ``amp``：采样循环套 fp16 autocast（Turing 及以上的 tensor core 提速
        约 1.5~2×）。输出会转回 fp32；质量差异在本模型的 64×64 输出上
        实测不可分辨（见 docs/LOTTERY.md 的对照记录）。
        """
        self.eval()
        if shape is None:
            shape = (self.in_ch, 64, 64)
        if x_T is None:
            x = torch.randn(n, *shape, device=device)
        else:
            x = x_T.to(device)
            n = x.size(0)
        use_cfg = cfg_scale != 1.0 and cond is not None and getattr(self.unet, "cond_dim", 0) > 0
        cond_null = torch.zeros_like(cond) if use_cfg else None
        step_ids = torch.linspace(self.T - 1, 0, ddim_steps).long().to(device)
        acp = self.alphas_cumprod
        use_amp = bool(amp) and torch.cuda.is_available() and str(device).startswith("cuda")
        ctx = (torch.autocast("cuda", dtype=torch.float16)
               if use_amp else contextlib.nullcontext())
        with ctx:
            for i, t in enumerate(step_ids):
                t_batch = t.expand(n).long()
                eps = self.unet(x, t_batch, cond, extra)
                if use_cfg:
                    eps_u = self.unet(x, t_batch, cond_null, extra)
                    eps = eps_u + cfg_scale * (eps - eps_u)
                a_t = acp[t]
                a_prev = acp[step_ids[i + 1]] if i + 1 < len(step_ids) else torch.tensor(1.0, device=device)
                x0 = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-1, 1)
                sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
                c = torch.sqrt(1 - a_prev - sigma ** 2) * eps
                x = torch.sqrt(a_prev) * x0 + c
                if sigma > 0 and i + 1 < len(step_ids):
                    x = x + sigma * torch.randn_like(x)
        return x.float() if use_amp else x


def drop_cond(cond: torch.Tensor | None, p: float) -> torch.Tensor | None:
    """条件 dropout：以概率 ``p`` 把整条样本的条件置零（逐样本，不按元素）。

    置零而不是传 ``None``：采样时 CFG 的无条件一侧也用零向量，
    两边必须走**同一条前向路径**（``cond_proj`` 的 bias、FiLM 都要参与），
    否则 CFG 的「无条件预测」不是模型训练时见过的那个分布。
    """
    if cond is None or p <= 0:
        return cond
    keep = (torch.rand(cond.size(0), 1, device=cond.device) >= p).to(cond.dtype)
    return cond * keep


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
