import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from fms.models.conformer import ConformerConfig, ConformerEncoder
from fms.models.granite import GraniteConfig, GraniteHeadless


@dataclass
class GraniteSpeechConfig:
    """
    Minimal wrapper config combining:
    - Conformer encoder
    - (Future) Speech Projector
    - Granite decoder
    """
    conformer_config: ConformerConfig
    projector_config: Optional[object] = None   # Placeholder
    decoder_config: GraniteConfig = None


class GraniteSpeech(nn.Module):
    """
    Minimal Granite-Speech wrapper (encoder + decoder only for now)
    Projector will be added later.
    """
    def __init__(
        self,
        conformer_config: ConformerConfig,
        projector_config: Optional[object],
        decoder_config: GraniteConfig,
        **kwargs,
    ):
        super().__init__()

        # Encoder
        self.encoder = ConformerEncoder(conformer_config)

        # TODO: Add projector when implemented
        self.projector = None

        # Decoder headless model (embeddings + transformer blocks)
        self.decoder = GraniteHeadless(decoder_config)

        # Final LM head
        self.lm_head = nn.Linear(
            decoder_config.emb_dim,
            decoder_config.src_vocab_size,
            bias=False
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        audio_features: (batch, seq_len, mel_bins)
        """
        # 1. Encoder
        hidden_states = self.encoder(audio_features)

        # 2. Projector (not implemented yet)
        if self.projector is not None:
            hidden_states = self.projector(hidden_states, attention_mask)

        # 3. Decoder (expects token embeddings OR embedded vectors)
        decoder_out, _ = self.decoder(hidden_states)

        # 4. LM Head -> logits
        logits = self.lm_head(decoder_out)

        return logits
