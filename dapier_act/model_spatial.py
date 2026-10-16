import torch
import torch.nn as nn

from dapier_act.model import DapierACT
from dapier_act.pointcloud_preprocess import (
    farthest_point_sample,
    linspace_candidates,
)


class SpatialPointTokenEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_tokens=256,
        candidate_points=8192,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.candidate_points = candidate_points

        self.feature_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )

        self.position_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def make_candidates(self, xyz):
        n = xyz.shape[0]

        if n <= self.candidate_points:
            return xyz

        indices = torch.linspace(
            0,
            n - 1,
            self.candidate_points,
            device=xyz.device,
        ).round().long()

        return xyz.index_select(
            0,
            indices,
        )

    def farthest_point_sample(self, xyz):
        n = xyz.shape[0]
        k = min(
            self.num_tokens,
            n,
        )

        selected = torch.empty(
            k,
            dtype=torch.long,
            device=xyz.device,
        )

        min_dist = torch.full(
            (n,),
            float("inf"),
            device=xyz.device,
        )

        center = xyz.mean(
            dim=0,
            keepdim=True,
        )

        farthest = torch.argmax(
            torch.sum(
                (xyz - center) ** 2,
                dim=1,
            )
        )

        for i in range(k):
            selected[i] = farthest

            point = xyz[
                farthest
            ].unsqueeze(0)

            dist = torch.sum(
                (xyz - point) ** 2,
                dim=1,
            )

            min_dist = torch.minimum(
                min_dist,
                dist,
            )

            farthest = torch.argmax(
                min_dist
            )

        sampled = xyz.index_select(
            0,
            selected,
        )

        if k < self.num_tokens:
            repeat_count = (
                self.num_tokens + k - 1
            ) // k

            sampled = sampled.repeat(
                repeat_count,
                1,
            )[:self.num_tokens]

        return sampled

    def sample_points(self, xyz):
        candidates = self.make_candidates(
            xyz
        )

        return self.farthest_point_sample(
            candidates
        )

    def forward(self, pointclouds):
        token_batches = []
        xyz_batches = []

        for cloud in pointclouds:
            if cloud.ndim != 2:
                raise ValueError(
                    f"Expected [N, C], got {tuple(cloud.shape)}"
                )

            if cloud.shape[1] < 3:
                raise ValueError(
                    f"Expected at least XYZ, got {tuple(cloud.shape)}"
                )

            xyz = cloud[:, :3]

            if xyz.shape[0] == 0:
                raise ValueError(
                    "Pointcloud contains zero points"
                )

            sampled_xyz = self.sample_points(
                xyz
            )

            features = self.feature_mlp(
                sampled_xyz
            )

            positions = self.position_mlp(
                sampled_xyz
            )

            tokens = self.norm(
                features + positions
            )

            token_batches.append(
                tokens
            )

            xyz_batches.append(
                sampled_xyz
            )

        return (
            torch.stack(
                token_batches,
                dim=0,
            ),
            torch.stack(
                xyz_batches,
                dim=0,
            ),
        )


class DP3PointGlobalEncoder(nn.Module):
    """DP3-inspired XYZ PointNet that emits one global 256-D token.

    Pipeline per sample:
        Task-frame cropped XYZ in meters
        -> candidate reduction
        -> FPS in metric space
        -> train-set limits normalization
        -> 3->64->128->hidden_dim with LayerNorm + ReLU
        -> symmetric max pool
        -> [1, hidden_dim] global token

    Unlike the original DP3 default head, the final 256->64 projection is
    intentionally omitted so the feature matches ACT's hidden_dim directly.
    """

    def __init__(
        self,
        hidden_dim=256,
        num_points=512,
        candidate_points=8192,
        point_min=None,
        point_max=None,
        range_eps=1e-4,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_points = int(num_points)
        self.candidate_points = int(candidate_points)
        self.range_eps = float(range_eps)

        if self.num_points <= 0:
            raise ValueError(
                f"num_points must be > 0, got {self.num_points}"
            )

        default_min = torch.zeros(3, dtype=torch.float32)
        default_max = torch.ones(3, dtype=torch.float32)

        if point_min is not None:
            default_min = torch.as_tensor(
                point_min,
                dtype=torch.float32,
            ).reshape(3)

        if point_max is not None:
            default_max = torch.as_tensor(
                point_max,
                dtype=torch.float32,
            ).reshape(3)

        self.register_buffer(
            "point_min",
            default_min.clone(),
        )
        self.register_buffer(
            "point_max",
            default_max.clone(),
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def set_point_limits(
        self,
        point_min,
        point_max,
    ):
        point_min = torch.as_tensor(
            point_min,
            dtype=self.point_min.dtype,
            device=self.point_min.device,
        ).reshape(3)

        point_max = torch.as_tensor(
            point_max,
            dtype=self.point_max.dtype,
            device=self.point_max.device,
        ).reshape(3)

        if torch.any(
            point_max - point_min < self.range_eps
        ):
            raise ValueError(
                "Degenerate point limits: "
                f"min={point_min.tolist()}, "
                f"max={point_max.tolist()}"
            )

        self.point_min.copy_(point_min)
        self.point_max.copy_(point_max)

    def sample_points(self, xyz):
        candidates = linspace_candidates(
            xyz,
            self.candidate_points,
        )

        return farthest_point_sample(
            candidates,
            self.num_points,
        )

    def normalize_points(self, xyz):
        point_range = (
            self.point_max - self.point_min
        ).clamp_min(
            self.range_eps
        )

        return (
            2.0
            * (xyz - self.point_min)
            / point_range
            - 1.0
        )

    def forward(self, pointclouds):
        token_batches = []
        xyz_batches = []

        for cloud in pointclouds:
            if cloud.ndim != 2:
                raise ValueError(
                    f"Expected [N, C], got {tuple(cloud.shape)}"
                )

            if cloud.shape[1] < 3:
                raise ValueError(
                    f"Expected at least XYZ, got {tuple(cloud.shape)}"
                )

            xyz = cloud[:, :3]

            if xyz.shape[0] == 0:
                raise ValueError(
                    "Pointcloud contains zero points"
                )

            sampled_xyz = self.sample_points(
                xyz
            )

            normalized_xyz = self.normalize_points(
                sampled_xyz
            )

            point_features = self.feature_mlp(
                normalized_xyz
            )

            global_feature = torch.max(
                point_features,
                dim=0,
            ).values

            token_batches.append(
                global_feature.unsqueeze(0)
            )

            xyz_batches.append(
                sampled_xyz
            )

        return (
            torch.stack(
                token_batches,
                dim=0,
            ),
            torch.stack(
                xyz_batches,
                dim=0,
            ),
        )


class DapierACTSpatial(DapierACT):
    def __init__(
        self,
        qpos_dim=12,
        action_dim=12,
        chunk_size=20,
        hidden_dim=256,
        latent_dim=32,
        num_point_tokens=256,
        candidate_points=8192,
        nheads=8,
        encoder_layers=4,
        decoder_layers=4,
        ff_dim=1024,
        dropout=0.0,
    ):
        super().__init__(
            qpos_dim=qpos_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_point_tokens=num_point_tokens,
            nheads=nheads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.point_encoder = SpatialPointTokenEncoder(
            hidden_dim=hidden_dim,
            num_tokens=num_point_tokens,
            candidate_points=candidate_points,
        )

        self.num_point_tokens = num_point_tokens
        self.candidate_points = candidate_points

    def encode_pointcloud(
        self,
        pointcloud,
    ):
        return self.point_encoder(
            pointcloud
        )

    def forward(
        self,
        qpos,
        pointcloud,
        actions=None,
        is_pad=None,
        return_point_features=False,
    ):
        batch_size = qpos.shape[0]

        if len(pointcloud) != batch_size:
            raise ValueError(
                f"Pointcloud batch size {len(pointcloud)} "
                f"does not match qpos batch size {batch_size}"
            )

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

            latent, mu, logvar = self.encode_latent(
                qpos,
                actions,
                is_pad,
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

        point_features, sampled_xyz = (
            self.encode_pointcloud(
                pointcloud
            )
        )

        point_tokens = (
            point_features
            + self.modality_embedding.weight[
                2
            ][None, None, :]
        )

        qpos_token = self.qpos_proj(
            qpos
        ).unsqueeze(1)

        qpos_token = (
            qpos_token
            + self.modality_embedding.weight[
                1
            ][None, None, :]
        )

        latent_token = self.latent_out_proj(
            latent
        ).unsqueeze(1)

        latent_token = (
            latent_token
            + self.modality_embedding.weight[
                0
            ][None, None, :]
        )

        context = torch.cat(
            [
                latent_token,
                qpos_token,
                point_tokens,
            ],
            dim=1,
        )

        memory = self.context_encoder(
            context
        )

        action_queries = (
            self.action_queries.weight
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        decoded = self.action_decoder(
            action_queries,
            memory,
        )

        action_hat = self.action_head(
            decoded
        )

        if return_point_features:
            return (
                action_hat,
                mu,
                logvar,
                {
                    "sampled_xyz": sampled_xyz,
                    "point_features": point_features,
                },
            )

        return (
            action_hat,
            mu,
            logvar,
        )
