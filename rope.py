from typing import Tuple
import torch
import numpy as np

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert ndim >= 4, f"Expected x to have at least 4 dimensions, but got {ndim}"
    assert freqs_cis.shape[0] == x.shape[1], f"Sequence length mismatch: {freqs_cis.shape[0]} vs {x.shape[1]}"
    assert freqs_cis.shape[1] == x.shape[-1] // 2, f"Dimension mismatch: {freqs_cis.shape[1]} vs {x.shape[-1] // 2}"
    
    shape = [1, x.shape[1], 1, x.shape[-1] // 2]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, seqlen, _, _ = query.shape
    device = query.device
    
    # Compute position-dependent rotation angles
    position = torch.arange(seqlen, device=device).unsqueeze(1)
    dim_t = torch.arange(0, head_dim, 2, device=device).float()
    freq = position / (theta ** (dim_t / head_dim))
    
    # Compute complex rotations
    freqs_cis = torch.polar(torch.ones_like(freq), freq)
    freqs_cis = reshape_for_broadcast(freqs_cis, query)

    # Reshape query and key
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Apply rotary embeddings
    query_out_real = query_real * freqs_cis.real - query_imag * freqs_cis.imag
    query_out_imag = query_real * freqs_cis.imag + query_imag * freqs_cis.real
    key_out_real = key_real * freqs_cis.real - key_imag * freqs_cis.imag
    key_out_imag = key_real * freqs_cis.imag + key_imag * freqs_cis.real

    query_out = torch.stack([query_out_real, query_out_imag], dim=-1).flatten(-2)
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1).flatten(-2)

    return query_out.type_as(query), key_out.type_as(key)
