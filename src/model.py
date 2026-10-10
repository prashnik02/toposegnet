"""
Baseline U-Net for CloudSEN12+ cloud segmentation.
Encoder: ResNet-34 pretrained on ImageNet
Input:   13 Sentinel-2 bands
Output:  4-class segmentation map (clear, thick, thin, shadow)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_baseline_unet(
    in_channels=13,
    num_classes=4,
    encoder_name="resnet34",
    pretrained=True
):
    """
    Returns a U-Net with pretrained ResNet-34 encoder.
    in_channels=13 because we have 13 Sentinel-2 bands.
    num_classes=4 because we have 4 cloud classes.
    """
    weights = "imagenet" if pretrained else None

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None   # raw logits — loss function handles softmax
    )
    return model


# ── Stage 5b(2): FiLM lat/lon conditioning ───
# Diagnostic that isolates INJECTION METHOD from feature choice. Stage 5b(1) fed
# lat/lon as 2 broadcast input channels (naive concat) -> test mIoU 0.6968, a
# mild regression. Here the SAME two scalars instead modulate the U-Net
# bottleneck via FiLM. If this beats 5b(1), the cause is the injection path, not
# the feature — the direct argument for TCM.
class LatLonFiLM(nn.Module):
    """
    Maps (lat, lon) -> per-channel (gamma, beta), then modulates a feature map:
        h <- (1 + gamma) * h + beta

    Zero-init on the last Linear makes gamma=0, beta=0 at step 0, so the block
    starts as an EXACT identity — the network at init is bit-for-bit the
    baseline U-Net. The conditioning must earn its influence via gradients,
    which removes any init-shock confound and makes a null result readable
    (gamma/beta simply stay near zero).
    """
    def __init__(self, cond_dim=2, feat_channels=512, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)   # gamma=beta=0 -> identity at init
        nn.init.zeros_(self.mlp[-1].bias)
        self.feat_channels = feat_channels

    def forward(self, h, cond):
        # h: (B, C, H, W) bottleneck features | cond: (B, 2) normalized lat/lon
        gamma, beta = self.mlp(cond).chunk(2, dim=1)   # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1) broadcast
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * h + beta


class FiLMUNet(nn.Module):
    """
    U-Net whose bottleneck is FiLM-conditioned on lat/lon.

    Input contract: takes the SAME (B, in_channels+2, H, W) tensor the
    `latlon_only` dataset mode already produces (13 bands + 2 constant lat/lon
    planes). The last 2 planes are constant by construction, so pixel (0,0)
    recovers the scalars exactly. Reusing that mode unchanged guarantees the
    lat/lon encoding is byte-identical to Stage 5b(1) — the feature is held
    fixed, only the injection path differs.

        x (B,15,512,512)
          ├─ bands = x[:, :13]        -> encoder -> feats[5] (B,512,16,16)
          └─ cond  = x[:, 13:, 0, 0]  -> FiLM ---^
                                          decoder (unchanged) -> (B,4,512,512)
    """
    def __init__(self, in_channels=13, num_classes=4,
                 encoder_name="resnet34", pretrained=True, hidden=64):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name    = encoder_name,
            encoder_weights = "imagenet" if pretrained else None,
            in_channels     = in_channels,
            classes         = num_classes,
            activation      = None,
        )
        bottleneck_ch   = self.unet.encoder.out_channels[-1]   # 512 for resnet34
        self.film       = LatLonFiLM(2, bottleneck_ch, hidden)
        self.in_channels = in_channels

    def forward(self, x):
        bands = x[:, :self.in_channels]                 # (B, 13, H, W)
        cond  = x[:, self.in_channels:, 0, 0]           # (B, 2) constant planes
        feats = list(self.unet.encoder(bands))          # list so feats[-1] is assignable
        feats[-1] = self.film(feats[-1], cond)          # condition the bottleneck only
        return self.unet.segmentation_head(self.unet.decoder(feats))


def build_film_unet(in_channels=13, num_classes=4,
                    encoder_name="resnet34", pretrained=True, hidden=64):
    """Stage 5b(2) model. in_channels = SPECTRAL bands only; lat/lon arrive via FiLM."""
    return FiLMUNet(in_channels, num_classes, encoder_name, pretrained, hidden)


# ── Stage 5b(3): MULTI-SCALE FiLM ────────────
# 5b(2) conditioned ONLY the bottleneck (feats[5]); every skip connection
# reached the decoder unconditioned, so the high-resolution path — which
# determines thin-cloud edges and shadow boundaries — carried no geo signal.
# This variant conditions feats[1..5]. NOTE the information source is unchanged
# (2 scalars): this adds capacity to EXPRESS lat/lon at multiple scales, it does
# not add information. Shared trunk + per-level heads so the lat/lon embedding
# is learned once and specialised per scale.
class MultiScaleFiLM(nn.Module):
    """
    cond (B,2) -> shared trunk -> per-level (gamma, beta) -> modulate each feature map.

    Every head is zero-init, so at step 0 EVERY level is an exact identity and
    the network is bit-for-bit the baseline U-Net — same guarantee as 5b(2),
    which keeps the two directly comparable.
    """
    def __init__(self, feat_channels, cond_dim=2, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 2 * c) for c in feat_channels])
        for h in self.heads:
            nn.init.zeros_(h.weight)   # gamma=beta=0 -> identity at init
            nn.init.zeros_(h.bias)
        self.feat_channels = list(feat_channels)

    def forward(self, feats, cond):
        """feats: list of (B,C_i,H_i,W_i) to condition, in the same order as feat_channels."""
        z = self.trunk(cond)
        out = []
        for f, head in zip(feats, self.heads):
            gamma, beta = head(z).chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            out.append((1.0 + gamma) * f + beta)
        return out


class MultiScaleFiLMUNet(nn.Module):
    """
    Same input contract as FiLMUNet (takes the unmodified 15ch latlon_only
    tensor and peels the 2 constant planes into scalars), but conditions
    feats[1..5] instead of feats[5] alone.

        x (B,15,512,512)
          |- bands = x[:, :13] -> encoder -> feats[0..5]
          |- cond  = x[:, 13:, 0, 0] (B,2)
                         |
             feats[1..5] all FiLM-modulated -> decoder -> (B,4,512,512)

    feats[0] is the raw input tensor (identity passthrough in smp) and is left
    alone — conditioning the input itself would just be a weird re-scaling.
    """
    def __init__(self, in_channels=13, num_classes=4,
                 encoder_name="resnet34", pretrained=True, hidden=64):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name    = encoder_name,
            encoder_weights = "imagenet" if pretrained else None,
            in_channels     = in_channels,
            classes         = num_classes,
            activation      = None,
        )
        # encoder.out_channels e.g. (13, 64, 64, 128, 256, 512); condition [1:]
        self.cond_levels = list(range(1, len(self.unet.encoder.out_channels)))
        self.film = MultiScaleFiLM(
            [self.unet.encoder.out_channels[i] for i in self.cond_levels], 2, hidden)
        self.in_channels = in_channels

    def forward(self, x):
        bands = x[:, :self.in_channels]
        cond  = x[:, self.in_channels:, 0, 0]              # (B,2)
        feats = list(self.unet.encoder(bands))
        modulated = self.film([feats[i] for i in self.cond_levels], cond)
        for slot, f in zip(self.cond_levels, modulated):
            feats[slot] = f
        return self.unet.segmentation_head(self.unet.decoder(feats))


def build_multiscale_film_unet(in_channels=13, num_classes=4,
                               encoder_name="resnet34", pretrained=True, hidden=64):
    """Stage 5b(3) model — FiLM at every encoder scale, not just the bottleneck."""
    return MultiScaleFiLMUNet(in_channels, num_classes, encoder_name, pretrained, hidden)


# ── Stage 6: TCM cross-attention ─────────────
# The paper's actual contribution. Everything before this conditioned on a
# terrain-derived signal via FiLM, i.e. a modulation that depends only on
# terrain and is applied regardless of image content. TCM instead lets the
# SPECTRAL features query the terrain, so the model can learn when and where
# terrain context matters. See src/tcm.py for the mechanism contrast.
class TerrainEncoder(nn.Module):
    """
    Tiny conv stack: elevation map -> features at the 5 U-Net encoder scales.

    Deliberately small (~98k params vs ResNet-34's ~21M). Terrain here is a
    CONDITIONING SIGNAL, not a second feature extractor — a heavy branch would
    make this a two-stream fusion net (GeoInfoNet's design, which the related
    work explicitly distinguishes TCM from).

    GroupNorm, not BatchNorm: batch is 4, and fresh BN statistics at that size
    are very noisy. The ResNet's own BN layers carry ImageNet running stats so
    they are fine; a new branch should not inherit that risk.
    """
    def __init__(self, in_ch=1, widths=(8, 16, 32, 64, 128)):
        super().__init__()
        self.widths = list(widths)
        blocks, c_prev = [], in_ch
        for c in widths:
            blocks.append(nn.Sequential(
                nn.Conv2d(c_prev, c, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(8, c), num_channels=c),
                nn.ReLU(inplace=True),
            ))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)

    def forward(self, dem):
        """dem: (B,1,512,512) -> [t1..t5] at 256,128,64,32,16 px."""
        feats, x = [], dem
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats


class TCMUNet(nn.Module):
    """
    U-Net whose two coarsest encoder features are cross-attention conditioned
    on elevation.

    Input contract mirrors FiLMUNet: takes the UNMODIFIED 14ch `elev` tensor
    and splits it, so the elevation feature is byte-identical to what Stage 2a
    consumed. That makes this a clean concat-vs-attention A/B on the same
    feature — the MSCloudCAM-style comparison the paper is built around.

        x (B,14,512,512)
          |- bands = x[:, :13] -> ResNet-34 -> feats[0..5]
          |- dem   = x[:, 13:14] -> TerrainEncoder -> t1..t5
                 feats[4] <- TCM(feats[4], t4)   256ch @ 32x32
                 feats[5] <- TCM(feats[5], t5)   512ch @ 16x16
                                     -> decoder -> (B,4,512,512)

    Only the two coarsest scales are conditioned: full HW x HW attention at
    64x64 needs ~1 GB forward (~3 GB with backward) which will not coexist with
    the U-Net on 6 GB. Windowed attention would be the follow-up if TCM shows
    an effect.

    Every TCM block is zero-init on its residual outputs, so the whole network
    is an EXACT identity to the 1a baseline at step 0 — the same guarantee the
    FiLM variants have, which is what keeps them comparable.
    """
    def __init__(self, in_channels=13, num_classes=4, encoder_name="resnet34",
                 pretrained=True, d_model=128, num_heads=4,
                 cond_levels=(4, 5), terrain_widths=(8, 16, 32, 64, 128),
                 input_size=512):
        super().__init__()
        from tcm import TerrainConditioningModule

        self.unet = smp.Unet(
            encoder_name    = encoder_name,
            encoder_weights = "imagenet" if pretrained else None,
            in_channels     = in_channels,
            classes         = num_classes,
            activation      = None,
        )
        self.in_channels  = in_channels
        self.cond_levels  = list(cond_levels)
        self.terrain      = TerrainEncoder(1, terrain_widths)

        enc_ch = self.unet.encoder.out_channels          # (13,64,64,128,256,512)
        self.tcm = nn.ModuleDict()
        for lvl in self.cond_levels:
            hw = input_size // (2 ** lvl)                # feats[l] is at 512/2^l
            self.tcm[str(lvl)] = TerrainConditioningModule(
                img_channels     = enc_ch[lvl],
                terrain_channels = terrain_widths[lvl - 1],   # t_l  <-> feats[l]
                d_model          = d_model,
                num_heads        = num_heads,
                spatial_size     = (hw, hw),
            )

    def forward(self, x):
        bands = x[:, :self.in_channels]
        dem   = x[:, self.in_channels:self.in_channels + 1]      # (B,1,H,W)

        feats = list(self.unet.encoder(bands))
        tfeats = self.terrain(dem)                                # t1..t5
        for lvl in self.cond_levels:
            feats[lvl] = self.tcm[str(lvl)](feats[lvl], tfeats[lvl - 1])
        return self.unet.segmentation_head(self.unet.decoder(feats))


class GAPTerrainModule(nn.Module):
    """
    TCM with the attention DELETED — the exact uniform-attention limit.


    Why this exists
    ---------------
    The attention-map analysis (2026-08-07) found TCM's attention is nearly
    uniform: normalised entropy 0.989 at feats[4] and 0.951 at feats[5], with
    the attended-elevation field flattened to ~0.14-0.17 of the terrain field's
    spatial SD. If attention is that diffuse, a global terrain summary should
    recover most of the gain — and if it does, the paper's contribution is
    "global terrain context helps", not "cross-attention helps".

    Why this is the RIGHT null, not merely a similar model
    -----------------------------------------------------
    TCM computes  out_i = sum_j A[i,j] * V_j   with  V = to_v(terrain).
    Set A[i,j] = 1/N (perfectly uniform) and that collapses to
        out_i = mean_j V_j = to_v(terrain).mean over space,
    because to_v is a 1x1 conv and therefore commutes with the spatial mean.
    This module computes exactly that. So it is not an approximation of the
    uniform-attention case — it IS the uniform-attention case, analytically.

    What is held constant vs TCM
    ----------------------------
      kept     : to_v, out_proj (zero-init), norm2, FFN (zero-init last conv),
                 both residual connections, d_model, ffn_ratio
      removed  : to_q, to_k, pos_k, norm1, and the softmax(QK^T) itself
    norm1 is dropped because it existed solely to produce Q; leaving it in would
    add unused parameters. Everything else is byte-for-byte the TCM design, so
    the ONLY difference is attention-weighted vs uniformly-averaged terrain.

    Parameter count is LOWER than TCM by construction. That is deliberate: if
    GAP matches TCM, the extra attention machinery is unjustified; if TCM wins,
    it wins while the cheaper variant had every other advantage held equal.
    """

    def __init__(self, img_channels, terrain_channels, d_model=128,
                 ffn_ratio=0.25):
        super().__init__()
        from tcm import ChannelLayerNorm

        # Terrain -> Value, identical to TCM's to_v
        self.to_v = nn.Conv2d(terrain_channels, d_model, 1, bias=False)

        # Back to image channel space. ZERO-INIT -> exact identity at step 0,
        # the same guarantee every other conditioning variant in this chain has.
        self.out_proj = nn.Conv2d(d_model, img_channels, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        self.norm2 = ChannelLayerNorm(img_channels)
        hidden = max(8, int(img_channels * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Conv2d(img_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, img_channels, 1),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, img_feat, terrain_feat):
        v = self.to_v(terrain_feat)                     # (B, d_model, H, W)
        v = v.mean(dim=(2, 3), keepdim=True)            # uniform attention == GAP
        img_feat = img_feat + self.out_proj(v)          # broadcast over H,W
        img_feat = img_feat + self.ffn(self.norm2(img_feat))
        return img_feat


class GAPTerrainUNet(nn.Module):
    """
    TCMUNet with every TCM block swapped for GAPTerrainModule.


    Input contract, conditioning levels, terrain encoder and zero-init identity
    are all identical to TCMUNet, so this is a drop-in A/B on the injection
    mechanism alone. Consumes the same 14ch `elev` tensor, meaning the feature
    is byte-identical to Stage 2a concat and Stage 6 TCM — three injection
    methods, one feature.
    """
    def __init__(self, in_channels=13, num_classes=4, encoder_name="resnet34",
                 pretrained=True, d_model=128, cond_levels=(4, 5),
                 terrain_widths=(8, 16, 32, 64, 128)):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name    = encoder_name,
            encoder_weights = "imagenet" if pretrained else None,
            in_channels     = in_channels,
            classes         = num_classes,
            activation      = None,
        )
        self.in_channels = in_channels
        self.cond_levels = list(cond_levels)
        self.terrain     = TerrainEncoder(1, terrain_widths)

        enc_ch = self.unet.encoder.out_channels          # (13,64,64,128,256,512)
        self.gap = nn.ModuleDict()
        for lvl in self.cond_levels:
            self.gap[str(lvl)] = GAPTerrainModule(
                img_channels     = enc_ch[lvl],
                terrain_channels = terrain_widths[lvl - 1],
                d_model          = d_model,
            )

    def forward(self, x):
        bands = x[:, :self.in_channels]
        dem   = x[:, self.in_channels:self.in_channels + 1]

        feats  = list(self.unet.encoder(bands))
        tfeats = self.terrain(dem)
        for lvl in self.cond_levels:
            feats[lvl] = self.gap[str(lvl)](feats[lvl], tfeats[lvl - 1])
        return self.unet.segmentation_head(self.unet.decoder(feats))


def build_gap_terrain_unet(in_channels=13, num_classes=4, encoder_name="resnet34",
                           pretrained=True, d_model=128, cond_levels=(4, 5)):
    """`gap_terrain` ablation: TCM with attention deleted. in_channels = BANDS only."""
    return GAPTerrainUNet(in_channels, num_classes, encoder_name, pretrained,
                          d_model, cond_levels)


def build_tcm_unet(in_channels=13, num_classes=4, encoder_name="resnet34",
                   pretrained=True, d_model=128, num_heads=4, cond_levels=(4, 5)):
    """Stage 6 model. in_channels = SPECTRAL bands only; elevation arrives via TCM."""
    return TCMUNet(in_channels, num_classes, encoder_name, pretrained,
                   d_model, num_heads, cond_levels)


# ── POOLED-ELEVATION FiLM CONTROL ────────────
# Why this exists: `elev_pooled` + arch="unet" was a NULL (0.6939 [0.6888,
# 0.7024], OVERLAP with everything, slope dose-response absent). But that null
# is WEAK: a channel that is constant across a patch enters the first conv as
# w*k + b with k fixed, so it can only act through that layer's effective bias
# — one scalar offset per output filter. It therefore cannot distinguish
# "pooling doesn't help" from "a concatenated channel cannot exploit pooling".
#
# This variant delivers the IDENTICAL scalar with depth-injection instead:
# gamma/beta modulation of the 512-ch bottleneck, the same route Stage 5b(2)
# used for lat/lon. If it recovers gap's gain, pooling was fine all along and
# the branch's contribution is depth-injection. If it is also null, the
# branch's extra capacity (128-d embedding at two levels) is what matters.
class FiLMElevUNet(nn.Module):
    """
    U-Net whose bottleneck is FiLM-conditioned on the PATCH-MEAN elevation.

    Input contract: takes the SAME (B, 14, H, W) tensor the `elev_pooled`
    dataset mode already produces (13 bands + 1 constant patch-mean elevation
    plane). That plane is constant by construction — the preflight verified
    within-patch spread is exactly 0.0 — so pixel (0,0) recovers the scalar
    exactly. Reusing the mode unchanged guarantees the elevation encoding is
    byte-identical to the concatenated control; the FEATURE is held fixed and
    only the injection path differs. This is the same discipline that made
    5b(1)-vs-5b(2) a clean comparison.

        x (B,14,512,512)
          ├─ bands = x[:, :13]        -> encoder -> feats[5] (B,512,16,16)
          └─ cond  = x[:, 13:, 0, 0]  -> FiLM ---^      (B,1) standardised metres
                                          decoder (unchanged) -> (B,4,512,512)

    Reuses LatLonFiLM verbatim with cond_dim=1 — the zero-init final Linear
    still makes gamma=beta=0 at step 0, so the net starts as an EXACT identity
    to the 13-band baseline and the conditioning has to be earned.
    """
    def __init__(self, in_channels=13, num_classes=4,
                 encoder_name="resnet34", pretrained=True, hidden=64):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name    = encoder_name,
            encoder_weights = "imagenet" if pretrained else None,
            in_channels     = in_channels,
            classes         = num_classes,
            activation      = None,
        )
        bottleneck_ch    = self.unet.encoder.out_channels[-1]   # 512 for resnet34
        self.film        = LatLonFiLM(1, bottleneck_ch, hidden)  # cond_dim=1: one scalar
        self.in_channels = in_channels

    def forward(self, x):
        bands = x[:, :self.in_channels]                 # (B, 13, H, W)
        cond  = x[:, self.in_channels:, 0, 0]           # (B, 1) constant plane
        feats = list(self.unet.encoder(bands))
        feats[-1] = self.film(feats[-1], cond)          # bottleneck only
        return self.unet.segmentation_head(self.unet.decoder(feats))


def build_film_elev_unet(in_channels=13, num_classes=4,
                         encoder_name="resnet34", pretrained=True, hidden=64):
    """Pooled-elevation FiLM control. in_channels = SPECTRAL bands only."""
    return FiLMElevUNet(in_channels, num_classes, encoder_name, pretrained, hidden)


# ── gap + lat/lon FiLM ───────────────────────
# Exploratory, single seed: does LOCATION add anything on top of TERRAIN once
# terrain is already injected at depth? Built on gap (the terminal model), not
# TCM, because TCM ≡ gap and the attention would be dead weight.
#
# Why FiLM and not attention for lat/lon: lat/lon is constant per patch. Added
# to the attention KEYS it shifts every logit of a query by the same q·c, which
# softmax cancels — attention is provably blind to it. Added to the VALUES it
# passes through as a constant offset, i.e. FiLM's beta without gamma. FiLM is
# the strictly more expressive route for a per-patch scalar.
class GAPTerrainLatLonUNet(GAPTerrainUNet):
    """
    GAPTerrainUNet plus a bottleneck LatLonFiLM.

        x (B,16,512,512)
          |- bands = x[:, :13]        -> ResNet-34 -> feats[0..5]
          |- dem   = x[:, 13:14]      -> TerrainEncoder -> gap @ feats[4], feats[5]
          |- cond  = x[:, 14:16, 0, 0] (B,2) lat/lon -> FiLM on feats[5] (after gap)
                                          -> decoder -> (B,4,512,512)

    Subclassing (film added AFTER super().__init__) means that under the same
    seed every gap parameter is initialised from the same RNG draws as a plain
    gap model; the zero-init FiLM then makes the whole net an EXACT identity to
    that gap model at step 0. The FiLM MLP is the only new parameter block.
    """
    def __init__(self, in_channels=13, num_classes=4, encoder_name="resnet34",
                 pretrained=True, d_model=128, cond_levels=(4, 5), hidden=64):
        super().__init__(in_channels, num_classes, encoder_name, pretrained,
                         d_model, cond_levels)
        self.film = LatLonFiLM(2, self.unet.encoder.out_channels[-1], hidden)

    def forward(self, x):
        bands = x[:, :self.in_channels]
        dem   = x[:, self.in_channels:self.in_channels + 1]
        cond  = x[:, self.in_channels + 1:self.in_channels + 3, 0, 0]   # (B,2) constant planes

        feats  = list(self.unet.encoder(bands))
        tfeats = self.terrain(dem)
        for lvl in self.cond_levels:
            feats[lvl] = self.gap[str(lvl)](feats[lvl], tfeats[lvl - 1])
        feats[-1] = self.film(feats[-1], cond)       # location modulates the terrain-conditioned bottleneck
        return self.unet.segmentation_head(self.unet.decoder(feats))


def build_gap_latlon_unet(in_channels=13, num_classes=4, encoder_name="resnet34",
                          pretrained=True, d_model=128, cond_levels=(4, 5), hidden=64):
    """gap + lat/lon bottleneck FiLM. in_channels = BANDS only (dataset gives 16)."""
    return GAPTerrainLatLonUNet(in_channels, num_classes, encoder_name, pretrained,
                                d_model, cond_levels, hidden)


if __name__ == "__main__":
    # Sanity check
    model = build_baseline_unet()
    model.eval()

    # Simulate one batch
    dummy_input = torch.randn(4, 13, 256, 256) # 4 Batch size
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")   # should be (4, 4, 256, 256)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model OK!")