# Copyright (c) Meta Platforms, Inc. and affiliates.
# MoGA: memory-object-conditioned gated-rank adaptation for SAM 2 memory attention

import math
from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.sam2_utils import get_activation_fn, get_clones



def sample_gumbel(shape, eps=1e-10, device='cpu'):
    """Sample from Gumbel distribution"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sigmoid(logits, tau=0.3, hard=False):
    """Gumbel-sigmoid function for differentiable discrete sampling"""
    noise = sample_gumbel(logits.shape, device=logits.device)
    y = (logits + noise) / tau
    y = torch.sigmoid(y)
    if not hard:
        return y
    else:
        y_hard = (y > 0.5).float()
        y_hard = (y_hard - y).detach() + y
        return y_hard

class MoGAGate(nn.Module):
    """MoGA gating MLP: shared across objects, applied per object pointer."""
    def __init__(self, d_model, rank, temperature=0.3):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.temperature = temperature

        # Project masklet to d_model for easier processing
        self.masklet_proj = nn.Linear(64, d_model)

        # Shared object corruption gate: From object-specific features
        self.object_gate = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, rank)
        )

    def forward(self, x, masklet):
        """Compute per-object gating logits from past-frame object-pointer features.

        SAM2 places each tracked object on the batch dimension B; past-frame object
        pointers are stacked along dim 0. So `masklet[t, b]` is the obj-pointer of
        object `b` from past-frame index `t`. The gate produced here is shared across
        all spatial positions N (frame-level decision) but separate per object (B).

        Args:
            x: [B, N, D] — current frame features
            masklet: [T, B, 64] when T past obj-pointer tokens are stored in memory,
                     or [B, 64] when only one is available.
        Returns:
            gate logits, shape [T, B, N, rank] (T>1) or [B, N, rank] (T==1).
        """
        B, N, D = x.shape

        if masklet.dim() == 3:  # [T_ptr, B, 64]
            num_ptr_tokens = masklet.shape[0]

            # One gate per past-frame pointer; per-object via B dim.
            gates = []
            for t in range(num_ptr_tokens):
                ptr_t = masklet[t]                                 # [B, 64]
                ptr_proj = self.masklet_proj(ptr_t)                # [B, D]
                gate_flat = self.object_gate(ptr_proj)             # [B, rank]
                gate = gate_flat.unsqueeze(1).expand(B, N, self.rank)  # [B, N, rank]
                gates.append(gate)

            return torch.stack(gates, dim=0)                       # [T_ptr, B, N, rank]

        elif masklet.dim() == 2:  # [B, 64]
            ptr_proj = self.masklet_proj(masklet)                  # [B, D]
            gate_flat = self.object_gate(ptr_proj)                 # [B, rank]
            return gate_flat.unsqueeze(1).expand(B, N, self.rank)  # [B, N, rank]
        else:
            raise ValueError(f"Unexpected masklet shape: {masklet.shape}")


class MoGALayer(nn.Module):
    """Memory-attention layer with MoGA adapters on self-attn Q/K and cross-attn Q."""

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
        low_rank: int = 16,
        high_rank: int = 256,
        lora_alpha: float = 1.0,
        temperature: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.low_rank = low_rank
        self.high_rank = high_rank
        self.lora_alpha = lora_alpha
        self.temperature = temperature

        # Object-only gates (shared across objects)
        self.sa_gate = MoGAGate(d_model, low_rank + high_rank, temperature)
        self.ca_gate = MoGAGate(d_model, low_rank + high_rank, temperature)

        # LoRA adapters (shared across objects)
        self.sa_lora_A = nn.Linear(d_model, low_rank + high_rank, bias=False)
        self.sa_lora_B = nn.Linear(low_rank + high_rank, d_model, bias=False)
        self.ca_lora_A = nn.Linear(d_model, low_rank + high_rank, bias=False)
        self.ca_lora_B = nn.Linear(low_rank + high_rank, d_model, bias=False)

        nn.init.kaiming_uniform_(self.sa_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.sa_lora_B.weight)
        nn.init.kaiming_uniform_(self.ca_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.ca_lora_B.weight)

    def _apply_moga(self, x, gate_module, lora_A, lora_B, masklet):
        """Apply the gated rank-1 adapter (MoGA) to current-frame features.

        Per-object gating is realized via the batch dimension B (SAM2 batches
        objects). When multiple past-frame object-pointer tokens are available,
        each one yields its own gating decision; the per-pointer LoRA outputs
        are averaged so that the rest of the memory-attention block sees a
        single [B, N, D] tensor (since lora_B is linear, this average is
        equivalent to averaging the gated rank-1 components first and applying
        lora_B once).

        Args:
            x: [B, N, D] — current frame features.
            masklet: [T_ptr, B, 64] or [B, 64] — past-frame obj-pointer features.
        Returns:
            x_adapted: [B, N, D] — x + LoRA update averaged over T_ptr (if any).
        """
        gate = gate_module(x, masklet)               # [T_ptr, B, N, rank] or [B, N, rank]
        A = lora_A(x)                                # [B, N, rank]; current-frame down-projection

        def _gate_and_apply(g):
            if self.training:
                m = gumbel_sigmoid(g, tau=self.temperature, hard=True)
            else:
                m = (torch.sigmoid(g) > 0.5).float()  # deterministic at eval (paper Eq. 6)
            gated_A = A * m                           # [B, N, rank] — per-rank-component selection
            return lora_B(gated_A) * (self.lora_alpha / lora_A.out_features)

        if gate.dim() == 4:                           # multiple past-frame obj-pointers
            # Average LoRA contributions across past-frame pointers.
            lora_outs = [_gate_and_apply(gate[t]) for t in range(gate.shape[0])]
            return x + torch.stack(lora_outs, dim=0).mean(dim=0)
        else:                                         # single obj-pointer
            return x + _gate_and_apply(gate)

    def _forward_sa(self, tgt, query_pos, masklet: Tensor):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        q = self._apply_moga(q, self.sa_gate, self.sa_lora_A, self.sa_lora_B, masklet)
        k = self._apply_moga(k, self.sa_gate, self.sa_lora_A, self.sa_lora_B, masklet)
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, masklet: Tensor, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        q = tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2
        q = self._apply_moga(q, self.ca_gate, self.ca_lora_A, self.ca_lora_B, masklet)

        tgt2 = self.cross_attn_image(
            q=q,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
        masklet: Optional[Tensor] = None,
    ) -> torch.Tensor:
        tgt = self._forward_sa(tgt, query_pos, masklet)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, masklet, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MoGA(nn.Module):
    """SAM 2 memory attention with MoGA layers."""

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        num_obj_ptr_tokens: int = 0,
        masklet: Optional[Tensor] = None,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (curr[0], curr_pos[0])

        assert curr.shape[1] == memory.shape[1], "Batch size must be the same"

        # Pull the past-frame object-pointer tokens out of the memory stack.
        # SAM2 builds `memory` (sam2_base._prepare_memory_conditioned_features)
        # as `[maskmem_tokens ..., obj_ptr_tokens]` along dim 0, so the last
        # `num_obj_ptr_tokens` positions are the object pointers m_o (paper Eq. 2).
        # dim 0 indexes past frames; objects live on the batch dim B.
        if masklet is None:
            if num_obj_ptr_tokens > 0:
                masklet = memory[-num_obj_ptr_tokens:]      # [T_ptr, B, 64]
            else:
                # No obj-pointer tokens yet (first frame): fall back to a
                # mean-pooled memory token as a single per-batch summary.
                masklet = memory.mean(dim=0)                # [B, 64]

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1) if curr_pos is not None else None
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1) if memory_pos is not None else None
            # Don't transpose masklet — keep as [T_ptr, B, 64].

        for layer in self.layers:
            if isinstance(layer, MoGALayer):
                output = layer(
                    tgt=output,
                    memory=memory,
                    masklet=masklet,
                    pos=memory_pos,
                    query_pos=curr_pos,
                    num_k_exclude_rope=num_obj_ptr_tokens,
                )

        normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)

        return normed_output

    def get_moga_parameters(self):
        """Extract MoGA adapters + LayerNorms for optimization.

        Per the paper's Implementation Details, the LayerNorms in the memory-
        attention block are co-trained with the MoGA adapter (LayerNorm tuning,
        following [De Min'23, Qi'22, ValizadehAslani'24, Zhao'23]).
        """
        params = []
        ln_count = 0

        # Top-level LayerNorm of the memory-attention block
        if isinstance(self.norm, nn.LayerNorm):
            params.extend(list(self.norm.parameters()))
            ln_count += 1

        for layer in self.layers:
            if isinstance(layer, MoGALayer):
                if hasattr(layer, 'sa_lora_A'):
                    params.extend([layer.sa_lora_A.weight, layer.sa_lora_B.weight])
                if hasattr(layer, 'ca_lora_A'):
                    params.extend([layer.ca_lora_A.weight, layer.ca_lora_B.weight])
                if hasattr(layer, 'sa_gate'):
                    params.extend(list(layer.sa_gate.parameters()))
                if hasattr(layer, 'ca_gate'):
                    params.extend(list(layer.ca_gate.parameters()))

                # Per-layer LayerNorms (norm1 / norm2 / norm3) — co-trained.
                for ln_name in ('norm1', 'norm2', 'norm3'):
                    ln = getattr(layer, ln_name, None)
                    if isinstance(ln, nn.LayerNorm):
                        params.extend(list(ln.parameters()))
                        ln_count += 1

        import logging
        logging.info(
            f"[MoGA] {len(params)} parameter tensors "
            f"({ln_count} LayerNorm layers co-trained)"
        )
        return params
