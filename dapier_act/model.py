import math

import torch
import torch.nn as nn


def sinusoidal_position_encoding(length, dim, device):
    position = torch.arange(
        length,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            dim,
            2,
            device=device,
            dtype=torch.float32,
        )
        * (-math.log(10000.0) / dim)
    )

    pe = torch.zeros(
        length,
        dim,
        device=device,
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class PointTokenEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_tokens=256,
    ):
        super().__init__()

        self.num_tokens = num_tokens

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

    def forward(self, xyz):
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(
                f"Expected pointcloud [B, N, 3], got {tuple(xyz.shape)}"
            )

        if xyz.shape[1] > self.num_tokens:
            indices = torch.linspace(
                0,
                xyz.shape[1] - 1,
                self.num_tokens,
                device=xyz.device,
            ).round().long()

            xyz = xyz.index_select(1, indices)

        features = self.feature_mlp(xyz)
        positions = self.position_mlp(xyz)

        return self.norm(features + positions)


class DapierACT(nn.Module):
    def __init__(
        self,
        qpos_dim=12,
        action_dim=12,
        chunk_size=20,
        hidden_dim=256,
        latent_dim=32,
        num_point_tokens=256,
        nheads=8,
        encoder_layers=4,
        decoder_layers=4,
        ff_dim=1024,
        dropout=0.0,
    ):
        super().__init__()

        self.qpos_dim = qpos_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.point_encoder = PointTokenEncoder(
            hidden_dim=hidden_dim,
            num_tokens=num_point_tokens,
        )

        self.qpos_proj = nn.Linear(
            qpos_dim,
            hidden_dim,
        )

        self.latent_out_proj = nn.Linear(
            latent_dim,
            hidden_dim,
        )

        self.modality_embedding = nn.Embedding(
            3,
            hidden_dim,
        )

        context_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.context_encoder = nn.TransformerEncoder(
            context_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.action_queries = nn.Embedding(
            chunk_size,
            hidden_dim,
        )

        self.action_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.cvae_cls = nn.Parameter(
            torch.zeros(
                1,
                1,
                hidden_dim,
            )
        )

        self.cvae_qpos_proj = nn.Linear(
            qpos_dim,
            hidden_dim,
        )

        self.cvae_action_proj = nn.Linear(
            action_dim,
            hidden_dim,
        )

        cvae_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.cvae_encoder = nn.TransformerEncoder(
            cvae_layer,
            num_layers=2,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.latent_stats = nn.Linear(
            hidden_dim,
            latent_dim * 2,
        )

        nn.init.normal_(
            self.cvae_cls,
            std=0.02,
        )

    def encode_latent(
        self,
        qpos,
        actions,
        is_pad,
    ):
        batch_size, sequence_length, _ = actions.shape

        cls_token = self.cvae_cls.expand(
            batch_size,
            -1,
            -1,
        )

        qpos_token = self.cvae_qpos_proj(
            qpos
        ).unsqueeze(1)

        action_tokens = self.cvae_action_proj(
            actions
        )

        tokens = torch.cat(
            [
                cls_token,
                qpos_token,
                action_tokens,
            ],
            dim=1,
        )

        positional_encoding = sinusoidal_position_encoding(
            sequence_length + 2,
            self.hidden_dim,
            tokens.device,
        ).unsqueeze(0)

        tokens = tokens + positional_encoding

        prefix_mask = torch.zeros(
            batch_size,
            2,
            dtype=torch.bool,
            device=tokens.device,
        )

        padding_mask = torch.cat(
            [
                prefix_mask,
                is_pad,
            ],
            dim=1,
        )

        encoded = self.cvae_encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        latent_stats = self.latent_stats(
            encoded[:, 0]
        )

        mu, logvar = latent_stats.chunk(
            2,
            dim=-1,
        )

        std = torch.exp(
            0.5 * logvar
        )

        latent = (
            mu
            + torch.randn_like(std) * std
        )

        return latent, mu, logvar

    def forward(
        self,
        qpos,
        pointcloud,
        actions=None,
        is_pad=None,
    ):
        batch_size = qpos.shape[0]

        if actions is not None:
            actions = actions[:, :self.chunk_size]

            if is_pad is None:
                is_pad = torch.zeros(
                    batch_size,
                    actions.shape[1],
                    dtype=torch.bool,
                    device=qpos.device,
                )
            else:
                is_pad = is_pad[:, :self.chunk_size]

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

        point_tokens = self.point_encoder(
            pointcloud
        )

        point_tokens = (
            point_tokens
            + self.modality_embedding.weight[2][None, None, :]
        )

        qpos_token = self.qpos_proj(
            qpos
        ).unsqueeze(1)

        qpos_token = (
            qpos_token
            + self.modality_embedding.weight[1][None, None, :]
        )

        latent_token = self.latent_out_proj(
            latent
        ).unsqueeze(1)

        latent_token = (
            latent_token
            + self.modality_embedding.weight[0][None, None, :]
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
            .expand(batch_size, -1, -1)
        )

        decoded = self.action_decoder(
            action_queries,
            memory,
        )

        action_hat = self.action_head(
            decoded
        )

        return action_hat, mu, logvar


def dapier_act_loss(
    action_hat,
    actions,
    is_pad,
    mu,
    logvar,
    kl_weight=0.01,
):
    valid = (
        ~is_pad
    ).unsqueeze(-1).to(
        action_hat.dtype
    )

    absolute_error = torch.abs(
        action_hat - actions
    )

    l1 = (
        absolute_error * valid
    ).sum()

    denominator = (
        valid.sum().clamp_min(1.0)
        * action_hat.shape[-1]
    )

    l1 = l1 / denominator

    if mu is None:
        kl = action_hat.new_tensor(0.0)
    else:
        kl = -0.5 * (
            1.0
            + logvar
            - mu.pow(2)
            - logvar.exp()
        )

        kl = kl.sum(
            dim=-1
        ).mean()

    total = (
        l1
        + kl_weight * kl
    )

    return {
        "loss": total,
        "l1": l1,
        "kl": kl,
    }
