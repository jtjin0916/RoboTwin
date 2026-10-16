import math

import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops.misc import FrozenBatchNorm2d

from dapier_act.model_spatial import (
    DapierACTSpatial,
    DP3PointGlobalEncoder,
)


# ============================================================
# 2D sine positional encoding
# - DETR / ACT style
# - No learnable parameters
# - Output: [B, H*W, hidden_dim]
# ============================================================
class PositionEmbeddingSine2D(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        temperature=10000.0,
        normalize=True,
        scale=2.0 * math.pi,
    ):
        super().__init__()

        if hidden_dim % 4 != 0:
            raise ValueError(
                f"hidden_dim must be divisible by 4, got {hidden_dim}"
            )

        self.hidden_dim = hidden_dim
        self.num_pos_feats = hidden_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, feature_map):
        """
        feature_map:
            [B, C, H, W]

        return:
            [B, H*W, hidden_dim]
        """
        b, _, h, w = feature_map.shape
        device = feature_map.device
        dtype = feature_map.dtype

        y_embed = torch.arange(
            1,
            h + 1,
            device=device,
            dtype=dtype,
        ).view(1, h, 1).expand(b, h, w)

        x_embed = torch.arange(
            1,
            w + 1,
            device=device,
            dtype=dtype,
        ).view(1, 1, w).expand(b, h, w)

        if self.normalize:
            eps = 1e-6

            y_embed = (
                y_embed
                / (y_embed[:, -1:, :] + eps)
                * self.scale
            )

            x_embed = (
                x_embed
                / (x_embed[:, :, -1:] + eps)
                * self.scale
            )

        dim_t = torch.arange(
            self.num_pos_feats,
            device=device,
            dtype=dtype,
        )

        dim_t = self.temperature ** (
            2
            * torch.div(
                dim_t,
                2,
                rounding_mode="floor",
            )
            / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (
                pos_x[:, :, :, 0::2].sin(),
                pos_x[:, :, :, 1::2].cos(),
            ),
            dim=4,
        ).flatten(3)

        pos_y = torch.stack(
            (
                pos_y[:, :, :, 0::2].sin(),
                pos_y[:, :, :, 1::2].cos(),
            ),
            dim=4,
        ).flatten(3)

        pos = torch.cat(
            (pos_y, pos_x),
            dim=3,
        )

        return pos.flatten(
            1,
            2,
        )


# ============================================================
# Independent wrist-camera ResNet18 encoder
#
# Original ACT style:
# RGB [0,1]
# → ResNet18 through layer4
# → no avgpool
# → no fc
# → 1x1 Conv 512 -> hidden_dim
# → spatial tokens
# ============================================================
class WristResNet18Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )

        # Keep ResNet only through layer4.
        # avgpool and fc are intentionally not used.
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # ACT / DETR-style feature projection.
        self.input_proj = nn.Conv2d(
            512,
            hidden_dim,
            kernel_size=1,
        )

        self.position_embedding = (
            PositionEmbeddingSine2D(
                hidden_dim=hidden_dim,
            )
        )

        self.hidden_dim = hidden_dim

    def forward(self, image):
        """
        image:
            [B, 3, H, W]
            RGB float tensor in [0, 1]

        For current RoboTwin wrist input:
            [B, 3, 240, 320]

        Expected layer4 feature:
            [B, 512, 8, 10]

        Projected:
            [B, 256, 8, 10]

        Tokens:
            [B, 80, 256]
        """
        if image.ndim != 4:
            raise ValueError(
                f"Expected wrist image [B,3,H,W], "
                f"got {tuple(image.shape)}"
            )

        if image.shape[1] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, "
                f"got {image.shape[1]}"
            )

        features = self.backbone(
            image
        )

        features = self.input_proj(
            features
        )

        # [B, C, H, W]
        # -> [B, C, H*W]
        # -> [B, H*W, C]
        tokens = (
            features
            .flatten(2)
            .transpose(1, 2)
        )

        positions = self.position_embedding(
            features
        )

        return (
            tokens,
            positions,
            features,
        )


# ============================================================
# DAPIER Multimodal ACT
# ============================================================
class DapierACTMultimodal(DapierACTSpatial):
    """
    Observation:
        Head PointCloud
        + Left Wrist RGB
        + Right Wrist RGB
        + qpos

    Head:
        DP3-inspired PointNet global encoder
        (Task-cropped XYZ -> FPS512 -> limits normalize -> max pool)

    Left / Right Wrist:
        Separate ImageNet-pretrained ResNet18
        Native 240x320
        layer4
        8x10 = 80 tokens / camera

    Context:
        1 head token (256D global PointCloud feature)
        + 80 left tokens
        + 80 right tokens
        = 161 visual tokens

    qpos and latent are appended as existing ACT context tokens.
    """

    HEAD_SOURCE = 0
    LEFT_SOURCE = 1
    RIGHT_SOURCE = 2

    def __init__(
        self,
        qpos_dim=12,
        action_dim=12,
        chunk_size=20,
        hidden_dim=256,
        latent_dim=32,
        num_point_samples=512,
        candidate_points=8192,
        point_min=None,
        point_max=None,
        nheads=8,
        encoder_layers=4,
        decoder_layers=4,
        ff_dim=1024,
        dropout=0.0,
        pretrained_wrist_backbone=True,
    ):
        super().__init__(
            qpos_dim=qpos_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            # The DP3 head emits exactly one global ACT token.
            num_point_tokens=1,
            candidate_points=candidate_points,
            nheads=nheads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Replace the inherited Spatial256 point-token encoder
        # with a DP3-inspired global PointNet encoder.
        #
        # Metric Task-frame XYZ -> candidate reduction -> FPS512
        # -> train-only limits normalization -> PointNet -> max pool
        # -> one 256-D Head token.
        # ----------------------------------------------------
        self.point_encoder = DP3PointGlobalEncoder(
            hidden_dim=hidden_dim,
            num_points=num_point_samples,
            candidate_points=candidate_points,
            point_min=point_min,
            point_max=point_max,
        )

        self.num_point_samples = int(
            num_point_samples
        )
        self.num_point_tokens = 1
        self.candidate_points = int(
            candidate_points
        )

        # ----------------------------------------------------
        # Completely independent wrist-camera encoders.
        # They start from the same ImageNet initialization,
        # but parameters are NOT shared.
        # ----------------------------------------------------
        self.left_wrist_encoder = (
            WristResNet18Encoder(
                hidden_dim=hidden_dim,
                pretrained=pretrained_wrist_backbone,
            )
        )

        self.right_wrist_encoder = (
            WristResNet18Encoder(
                hidden_dim=hidden_dim,
                pretrained=pretrained_wrist_backbone,
            )
        )

        # ----------------------------------------------------
        # Explicit sensor/source identity.
        #
        # 0 = Head PointCloud
        # 1 = Left Wrist RGB
        # 2 = Right Wrist RGB
        #
        # Each is a learnable hidden_dim-dimensional vector.
        # No extra feature dimension is created.
        # ----------------------------------------------------
        self.source_embedding = nn.Embedding(
            3,
            hidden_dim,
        )

        nn.init.normal_(
            self.source_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        self.hidden_dim = hidden_dim

    def encode_wrist(
        self,
        image,
        encoder,
        source_id,
    ):
        """
        image:
            [B,3,H,W], float RGB [0,1]

        output:
            tokens [B, HW, hidden_dim]
        """
        (
            tokens,
            positions,
            feature_map,
        ) = encoder(
            image
        )

        source = self.source_embedding.weight[
            source_id
        ][None, None, :]

        # feature
        # + 2D image position
        # + camera/source identity
        tokens = (
            tokens
            + positions
            + source
        )

        return (
            tokens,
            feature_map,
        )

    def forward(
        self,
        qpos,
        pointcloud,
        left_wrist,
        right_wrist,
        actions=None,
        is_pad=None,
        return_features=False,
    ):
        batch_size = qpos.shape[0]

        # ----------------------------------------------------
        # Basic batch validation
        # ----------------------------------------------------
        if len(pointcloud) != batch_size:
            raise ValueError(
                f"Pointcloud batch size {len(pointcloud)} "
                f"does not match qpos batch size {batch_size}"
            )

        if left_wrist.shape[0] != batch_size:
            raise ValueError(
                f"Left wrist batch size "
                f"{left_wrist.shape[0]} "
                f"does not match qpos batch size "
                f"{batch_size}"
            )

        if right_wrist.shape[0] != batch_size:
            raise ValueError(
                f"Right wrist batch size "
                f"{right_wrist.shape[0]} "
                f"does not match qpos batch size "
                f"{batch_size}"
            )

        # ----------------------------------------------------
        # CVAE latent
        #
        # Train:
        #   qpos + GT action chunk -> z
        #
        # Inference:
        #   z = 0
        # ----------------------------------------------------
        if actions is not None:
            actions = actions[
                :,
                :self.chunk_size,
            ]

            if is_pad is None:
                is_pad = torch.zeros(
                    batch_size,
                    actions.shape[1],
                    dtype=torch.bool,
                    device=qpos.device,
                )
            else:
                is_pad = is_pad[
                    :,
                    :self.chunk_size,
                ]

            latent, mu, logvar = (
                self.encode_latent(
                    qpos,
                    actions,
                    is_pad,
                )
            )

        else:
            latent = torch.zeros(
                batch_size,
                self.latent_dim,
                dtype=qpos.dtype,
                device=qpos.device,
            )

            mu = None
            logvar = None

        # ----------------------------------------------------
        # 1. Head PointCloud -> DP3-inspired global 256D token
        # ----------------------------------------------------
        (
            point_features,
            sampled_xyz,
        ) = self.encode_pointcloud(
            pointcloud
        )

        # Keep the existing point-modality embedding and add explicit
        # HEAD sensor identity to the single global point token.
        head_tokens = (
            point_features
            + self.modality_embedding.weight[
                2
            ][None, None, :]
            + self.source_embedding.weight[
                self.HEAD_SOURCE
            ][None, None, :]
        )

        # ----------------------------------------------------
        # 2. Left Wrist RGB
        # ----------------------------------------------------
        (
            left_tokens,
            left_feature_map,
        ) = self.encode_wrist(
            left_wrist,
            self.left_wrist_encoder,
            self.LEFT_SOURCE,
        )

        # ----------------------------------------------------
        # 3. Right Wrist RGB
        # ----------------------------------------------------
        (
            right_tokens,
            right_feature_map,
        ) = self.encode_wrist(
            right_wrist,
            self.right_wrist_encoder,
            self.RIGHT_SOURCE,
        )

        # ----------------------------------------------------
        # Existing qpos token
        # ----------------------------------------------------
        qpos_token = self.qpos_proj(
            qpos
        ).unsqueeze(1)

        qpos_token = (
            qpos_token
            + self.modality_embedding.weight[
                1
            ][None, None, :]
        )

        # ----------------------------------------------------
        # Existing latent token
        # ----------------------------------------------------
        latent_token = self.latent_out_proj(
            latent
        ).unsqueeze(1)

        latent_token = (
            latent_token
            + self.modality_embedding.weight[
                0
            ][None, None, :]
        )

        # ----------------------------------------------------
        # Multimodal context
        #
        # Expected with native 240x320 wrists:
        #
        # latent       1
        # qpos         1
        # head         1
        # left        80
        # right       80
        # ----------------
        # total      163 tokens
        # ----------------------------------------------------
        context = torch.cat(
            [
                latent_token,
                qpos_token,
                head_tokens,
                left_tokens,
                right_tokens,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Existing ACT Transformer encoder
        # ----------------------------------------------------
        memory = self.context_encoder(
            context
        )

        # ----------------------------------------------------
        # Existing 20 learned action queries
        # ----------------------------------------------------
        action_queries = (
            self.action_queries.weight
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        # ----------------------------------------------------
        # Existing ACT decoder
        # ----------------------------------------------------
        decoded = self.action_decoder(
            action_queries,
            memory,
        )

        action_hat = self.action_head(
            decoded
        )

        if return_features:
            return (
                action_hat,
                mu,
                logvar,
                {
                    "sampled_xyz": sampled_xyz,
                    "point_features": point_features,
                    "head_tokens": head_tokens,
                    "left_tokens": left_tokens,
                    "right_tokens": right_tokens,
                    "left_feature_map": left_feature_map,
                    "right_feature_map": right_feature_map,
                    "context": context,
                },
            )

        return (
            action_hat,
            mu,
            logvar,
        )
