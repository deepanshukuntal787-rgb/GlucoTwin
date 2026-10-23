"""FusionTwinNet: Dual-stream multimodal physiological digital twin network."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from gluco_twin.config import cfg

class StaticEncoder(nn.Module):
    """Encodes static patient EHR features (demographics, disease, meds, labs)."""
    def __init__(self, in_dim: int = 21, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantileHead(nn.Module):
    """Outputs monotonic non-crossing quantiles: P10 <= P50 <= P90."""
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(in_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, 64)
        raw = self.fc(x)  # (B, H, 3)
        p50 = raw[..., 0]
        # Non-crossing guarantee: P10 = P50 - softplus(delta_10), P90 = P50 + softplus(delta_90)
        p10 = p50 - F.softplus(raw[..., 1])
        p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([p10, p50, p90], dim=-1)  # (B, H, 3)


class FusionTwinNet(nn.Module):
    """Multimodal digital twin forecasting and event prediction model."""
    def __init__(
        self,
        static_dim: int = cfg.static_dim,
        dynamic_dim: int = cfg.dynamic_dim,
        covariate_dim: int = cfg.covariate_dim,
        hidden_dim: int = cfg.hidden_dim,
        decoder_dim: int = cfg.decoder_dim,
        num_gru_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.decoder_dim = decoder_dim

        # 1. Static Stream
        self.static_encoder = StaticEncoder(static_dim, 64)
        self.h0_proj = nn.Linear(64, num_gru_layers * hidden_dim)

        # 2. Dynamic Stream: Conv1d stem + GRU
        self.conv_stem = nn.Sequential(
            nn.Conv1d(dynamic_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=cfg.dropout if num_gru_layers > 1 else 0.0,
        )
        self.temporal_summary = nn.Linear(hidden_dim * 2, hidden_dim)

        # 3. Fusion Layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # 4. Decoder (Future Covariates)
        self.decoder_init = nn.Linear(hidden_dim, decoder_dim)
        self.decoder_gru = nn.GRU(
            input_size=covariate_dim,
            hidden_size=decoder_dim,
            num_layers=1,
            batch_first=True,
        )
        self.quantile_head = QuantileHead(decoder_dim)

        # 5. Adverse Event Head
        self.event_head = nn.Sequential(
            nn.Linear(hidden_dim + decoder_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        dynamic_x: torch.Tensor,
        future_cov: torch.Tensor,
        static_x: torch.Tensor,
        adapter_b: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_shift: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        Args:
            dynamic_x: (B, 96, 16)
            future_cov: (B, 8, 5)
            static_x: (B, 21)
            adapter_b: (B, 128) optional per-patient adapter bias
        Returns:
            dict with:
                'quantiles': (B, 8, 3) [p10, p50, p90]
                'event_logit': (B, 1)
                'fused_state': (B, 128)
        """
        B = dynamic_x.size(0)

        # 1. Static embedding
        z_s = self.static_encoder(static_x)  # (B, 64)
        h0 = self.h0_proj(z_s)  # (B, num_gru_layers * hidden_dim)
        h0 = h0.view(B, self.num_gru_layers, self.hidden_dim).transpose(0, 1).contiguous()

        # 2. Dynamic Conv1d + GRU
        # dynamic_x: (B, T, C) -> Conv1d needs (B, C, T)
        stem_out = self.conv_stem(dynamic_x.transpose(1, 2)).transpose(1, 2)  # (B, 96, 64)
        gru_out, _ = self.gru(stem_out, h0)  # (B, 96, 128)

        # Summary = concat(last_step, mean_last_8)
        last_step = gru_out[:, -1, :]  # (B, 128)
        mean_last_8 = gru_out[:, -8:, :].mean(dim=1)  # (B, 128)
        summary = self.temporal_summary(torch.cat([last_step, mean_last_8], dim=-1))  # (B, 128)

        # 3. Fusion
        f = self.fusion_fc(torch.cat([summary, z_s], dim=-1))  # (B, 128)

        # Optional Patient Personalization Adapter
        if adapter_b is not None:
            f = f + adapter_b

        # 4. Decoder with Future Covariates
        dec_h0 = self.decoder_init(f).unsqueeze(0).contiguous()  # (1, B, 64)
        dec_out, _ = self.decoder_gru(future_cov, dec_h0)  # (B, 8, 64)
        quantiles = self.quantile_head(dec_out)  # (B, 8, 3)

        if output_scale is not None and output_shift is not None:
            quantiles = quantiles * output_scale + output_shift

        # 5. Event Head
        dec_mean = dec_out.mean(dim=1)  # (B, 64)
        event_feat = torch.cat([f, dec_mean], dim=-1)  # (B, 192)
        event_logit = self.event_head(event_feat)  # (B, 1)

        return {
            "quantiles": quantiles,
            "event_logit": event_logit,
            "fused_state": f,
            "decoder_feat": dec_out,
        }
