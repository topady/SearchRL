# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pure PyTorch implementations of flash_attn.bert_padding functions.
This allows the codebase to run without the flash-attn package installed.
"""

import torch
import torch.nn.functional as F


def unpad_input(hidden_states, attention_mask):
    """Remove padding from input sequences.

    Args:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool where True = keep (non-padding)

    Returns:
        unpadded_hidden_states: (total_nnz, ...)
        indices: (total_nnz,) flattened indices of non-padding tokens
        cu_seqlens: (batch + 1,) cumulative sequence lengths
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    hidden_states = hidden_states.view(-1, *hidden_states.shape[2:])
    unpadded = hidden_states[indices]
    return unpadded, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states, indices, batch, seqlen):
    """Re-pad unpadded sequences back to (batch, seqlen, ...).

    Args:
        hidden_states: (total_nnz, ...)
        indices: (total_nnz,) flattened indices returned by unpad_input
        batch: int, original batch size
        seqlen: int, original sequence length

    Returns:
        padded: (batch, seqlen, ...)
    """
    output = torch.zeros(batch * seqlen, *hidden_states.shape[1:],
                         dtype=hidden_states.dtype, device=hidden_states.device)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *hidden_states.shape[1:])


def index_first_axis(tensor, indices):
    """Gather elements along the first axis.

    Args:
        tensor: (total, ...)
        indices: (num_selected,)

    Returns:
        gathered: (num_selected, ...)
    """
    return tensor[indices]
