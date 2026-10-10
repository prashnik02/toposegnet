import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) map."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, HW, C)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(B, C, H, W)


class TerrainConditioningModule(nn.Module):
    """
    Terrain Conditioning Module (TCM)

    Cross-attention where terrain conditions spectral features: the image asks
    "what am I looking at?", terrain answers "here is the context at every
    location", and attention decides WHERE that context matters.

    Contrast with the FiLM baselines in model.py:
      FiLM      out = (1+gamma(terrain)) * h + beta(terrain)
                -> modulation depends ONLY on terrain, applied regardless of
                   what the image says, and is strictly per-pixel local.
      TCM       attn = softmax(Q(image) K(terrain)^T / sqrt(d))
                -> weighting depends on the INTERACTION of both, and pixel i
                   can pull terrain evidence from ANY pixel j (non-local).
    That difference is the paper's claim; everything else is held constant.

    ── REWRITTEN 2026-07-29 ────────────────────────────────────
    Three changes vs the original draft. The cross-attention maths is
    unchanged; what changed is normalisation placement, initialisation, and
    the attention kernel.

    (1) PRE-NORM instead of post-norm. The original did
            out = img_feat + out_proj(...);  out = LayerNorm(out)
        which can NEVER be an identity at init, because LayerNorm(img_feat)
        != img_feat even when the attention branch outputs exactly zero.
        Every other conditioning variant in this chain (5b(2) FiLM, 5b(3)
        multi-scale FiLM) starts as a bit-for-bit copy of the 1a baseline;
        losing that here would reintroduce init-shock as a confound and break
        comparability. Pre-norm also trains more stably than post-norm.

    (2) ZERO-INIT on out_proj and on the last FFN conv, so at step 0 both
        residual branches contribute exactly 0 and forward(x) == x. The
        conditioning has to be EARNED by gradient descent, which also makes a
        null result readable (weights stay near zero = model ignored terrain).

    (3) F.scaled_dot_product_attention instead of an explicit softmax(QK^T),
        which avoids materialising the (HW x HW) matrix. On a 6 GB card the
        explicit form costs ~1 GB forward at 64x64 and OOMs during backward.

    Args:
        img_channels     : channels in the image feature map
        terrain_channels : channels in the terrain feature map
        d_model          : attention width (default 128; 256 doubles params
                           for no demonstrated gain and invites the "it just
                           had more capacity" objection)
        num_heads        : attention heads (default 4)
        spatial_size     : (H, W) of this scale. Enables a learned positional
                           embedding on K — without it attention is
                           permutation-invariant over terrain tokens and can
                           match on content but never on PROXIMITY, which
                           matters for "is this pixel near high terrain".
        ffn_ratio        : FFN hidden width as a fraction of img_channels.
                           0.25 keeps the block cheap; the original 2.0 made
                           the FFN ~75% of all TCM parameters.
    """

    def __init__(self, img_channels, terrain_channels, d_model=128,
                 num_heads=4, spatial_size=None, ffn_ratio=0.25):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide by num_heads"

        self.num_heads = num_heads
        self.d_model   = d_model

        self.norm1 = ChannelLayerNorm(img_channels)     # pre-norm on the query side
        self.norm2 = ChannelLayerNorm(img_channels)     # pre-norm before the FFN

        # Image features -> Query.  "What does this region look like?"
        self.to_q = nn.Conv2d(img_channels, d_model, 1, bias=False)
        # Terrain features -> Key.  "What terrain sits at each location?"
        self.to_k = nn.Conv2d(terrain_channels, d_model, 1, bias=False)
        # Terrain features -> Value. "What should terrain contribute?"
        self.to_v = nn.Conv2d(terrain_channels, d_model, 1, bias=False)

        # Learned positional embedding added to K, so attention can express
        # spatial relationships and not just feature similarity.
        if spatial_size is not None:
            H, W = spatial_size
            self.pos_k = nn.Parameter(torch.zeros(1, d_model, H, W))
            nn.init.trunc_normal_(self.pos_k, std=0.02)
        else:
            self.pos_k = None

        # Back to image channel space. ZERO-INIT -> identity at step 0.
        self.out_proj = nn.Conv2d(d_model, img_channels, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        hidden = max(8, int(img_channels * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Conv2d(img_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, img_channels, 1),
        )
        nn.init.zeros_(self.ffn[-1].weight)             # ZERO-INIT
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, img_feat, terrain_feat):
        """
        img_feat     : (B, img_channels, H, W)
        terrain_feat : (B, terrain_channels, H, W)   same H, W
        returns      : (B, img_channels, H, W)
        """
        B, C, H, W = img_feat.shape

        # ── attention branch (pre-norm) ──────────────────────────────────
        q_in = self.norm1(img_feat)
        Q = self.to_q(q_in)
        K = self.to_k(terrain_feat)
        V = self.to_v(terrain_feat)

        if self.pos_k is not None:
            K = K + self.pos_k

        # (B, d_model, H, W) -> (B, heads, HW, head_dim)
        Q = rearrange(Q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        K = rearrange(K, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        V = rearrange(V, 'b (h d) x y -> b h (x y) d', h=self.num_heads)

        # flash/mem-efficient attention; scaling by 1/sqrt(head_dim) is internal
        out = F.scaled_dot_product_attention(Q, K, V)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)

        img_feat = img_feat + self.out_proj(out)        # exact identity at init

        # ── feed-forward branch (pre-norm) ───────────────────────────────
        img_feat = img_feat + self.ffn(self.norm2(img_feat))
        return img_feat


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing TCM on: {device}")

    # the two scales actually used by TCMUNet (see model.py)
    for img_ch, ter_ch, H in [(256, 64, 32), (512, 128, 16)]:
        img = torch.randn(2, img_ch, H, H, device=device)
        ter = torch.randn(2, ter_ch, H, H, device=device)
        tcm = TerrainConditioningModule(img_ch, ter_ch, d_model=128,
                                        num_heads=4, spatial_size=(H, H)).to(device)
        out = tcm(img, ter)
        assert out.shape == img.shape
        ident = (out - img).abs().max().item()
        n = sum(p.numel() for p in tcm.parameters())
        print(f"  {img_ch:4d}ch @ {H:2d}x{H:<2d} -> {tuple(out.shape)} | "
              f"identity at init {ident:.2e} | {n:,} params")
        assert ident == 0.0, "zero-init identity broken"

    print("\nAll TCM tests passed (shapes + exact zero-init identity).")
