from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn as nn
from transformers import AutoModel

from src.config import Config

class PCLClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        head_dropout: float,
        use_coral_objective: bool,
        use_type_multilabel_obj: bool,
        use_linear_head: bool = True
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        
        hidden_dim = self.encoder.config.hidden_size
        assert isinstance(hidden_dim, int)

        head_dim = 4 if use_coral_objective else 1

        if use_linear_head:
            self.main_head = nn.Sequential(
                nn.Dropout(head_dropout), 
                nn.Linear(hidden_dim, head_dim)
            )
        else:
            self.main_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(head_dropout), 
                nn.Linear(hidden_dim, head_dim)
            )
        
        self.type_multilabel_head = None
        if use_type_multilabel_obj:
            self.type_multilabel_head = nn.Sequential(
                nn.Dropout(head_dropout), nn.Linear(hidden_dim, 7)
            )

    def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0].to(torch.float32)

        logits = self.main_head(pooled)
        if self.type_multilabel_head is not None:
            aux_logits = self.type_multilabel_head(pooled)
            
            return logits, aux_logits

        return logits, None
    
    @classmethod
    def from_cfg(cls, cfg: Config) -> PCLClassifier:
        return cls(
            encoder_name=cfg.model.encoder_name,
            head_dropout=cfg.model.head_dropout,
            use_coral_objective=cfg.loss.use_coral_objective,
            use_type_multilabel_obj=cfg.loss.aux_multilabel_plc_loss_weight > 0,
            use_linear_head = cfg.model.use_linear_head
        )
