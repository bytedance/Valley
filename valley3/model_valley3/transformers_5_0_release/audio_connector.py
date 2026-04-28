# coding=utf-8

"""
AudioConnector: projection and optional downsampling for audio sequences prior to feeding the LLM trunk.

Target behavior:
- Input  : [B, T, C_in] audio features (mel or upstream encoder outputs) and per-sample valid lengths
- Output : [B, T', hidden_size] projected features and boolean padding mask [B, T'] with True at valid tokens
- Options: optional downsampling by a factor of 2; simple segment packing utilities
"""

from typing import Optional, Tuple

import torch
from torch import nn

class AudioConnector(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 5120,
        hidden_dim: int = 8192, # 4096,
        dropout: float = 0.1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # self.dropout_p = dropout

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        # self.dropout = nn.Dropout(p=dropout)
        self.proj_mid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim]
        Returns:
            out: [N, output_dim]
        """
        if x.ndim != 2:
            raise ValueError(
                f"AudioConnectorMLP only supports 2D input [N, C], but received shape {tuple(x.shape)}; "
                "please flatten to 2D in the upstream process before passing it in."
            )
        out = self.proj_in(x)
        out = self.act(out)
        out = self.proj_mid(out)
        out = self.act(out)
        # out = self.dropout(out)
        out = self.proj_out(out)
        out = self.ln(out)
        return out
