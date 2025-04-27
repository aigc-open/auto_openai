import torch
from typing import Union, Tuple, Optional
import importlib
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm


def new_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D]
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # # # used for lumina

        # Reshape and separate the real and imaginary parts
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]

        # Separate the real and imaginary parts of the frequencies
        freqs_cis_real = freqs_cis.unsqueeze(2).real
        freqs_cis_imag = freqs_cis.unsqueeze(2).imag

        # Compute the real and imaginary parts of the output
        real_part = x_real * freqs_cis_real - x_imag * freqs_cis_imag
        imag_part = x_real * freqs_cis_imag + x_imag * freqs_cis_real

        # Combine the real and imaginary parts into one tensor
        x_out_real_imag = torch.stack((real_part, imag_part), dim=-1)

        # Flatten the output to match the original structure
        x_out = x_out_real_imag.flatten(3)
        return x_out


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                # Split hidden states into real and imaginary parts
                hidden_shape = hidden_states.shape
                hidden_states_reshaped = hidden_states.unflatten(3, (-1, 2))
                real = hidden_states_reshaped[..., 0]
                imag = hidden_states_reshaped[..., 1]
                
                # Split frequencies into real and imaginary parts
                freqs_real = freqs.real
                freqs_imag = freqs.imag
                
                # Calculate the real and imaginary parts of the result
                # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                real_output = real * freqs_real - imag * freqs_imag
                imag_output = real * freqs_imag + imag * freqs_real
                
                # Combine and reshape to original format
                result = torch.stack([real_output, imag_output], dim=-1).flatten(3)
                
                return result.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def modify_apply_rotary_emb():
    """Modify the apply_rotary_emb function in memory without updating the source file."""
    # Dynamically import the diffusers module
    # diffusers_module = importlib.import_module('diffusers')

    # import the embeddings module
    embeddings_module = importlib.import_module('diffusers.models.embeddings')

    # Replace the old function with the new one
    embeddings_module.apply_rotary_emb = new_apply_rotary_emb
    # from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0

    transformer_wan = importlib.import_module('diffusers.models.transformers.transformer_wan')
    transformer_wan.WanAttnProcessor2_0 = WanAttnProcessor2_0

