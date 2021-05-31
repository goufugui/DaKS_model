import torch
import numpy as np
import torch.functional as F
import torch.nn as nn


class tattention(nn.Module):
  """Scaled dot-product attention mechanism."""
  def __init__(self, attention_dropout=0.0):
    """
    Init.
    Args:
      attention_dropout: A scalar, dropout rate.
    """
    super(tattention, self).__init__()
    self.dropout = nn.Dropout(attention_dropout)
    self.softmax = nn.Softmax(dim=-1)
    self.linear1 = nn.Linear(600, 600)
    self.w1 = nn.Linear(600, 600)
    self.w2 = nn.Linear(600, 600)
    self.layer_norm = nn.LayerNorm(600)
    self.relu = nn.ReLU(inplace=True)
  def forward(self,q, k, v, scale=None, attn_mask=None):
    """Forward pass.
    Args:
      q: Queries tensor, with shape of [B, L_q, D_q]
      k: Keys tensor, with shape of [B, L_k, D_k]
      v: Values tensor, with shape of [B, L_v, D_v]
      scale: A scalar, scale factor.
      attn_mask: A binary masking tensor, with shape of [B, L_q, L_k]
    Returns:
      Context and attention tensor.
    """
    attention = torch.bmm(q, k.transpose(1, 2))
    scale = (k.size(-1) // 1) ** -0.5
    if scale:
        attention = attention * scale
    if attn_mask is not None:
        # Mask out attention
        # set a negative infnite to where were padded a `PAD`
        attention = attention.masked_fill_(attn_mask, -np.inf)
    attention = self.softmax(attention)
    #attention = self.Dropout(attention)
    context = torch.bmm(attention, v)
    output = self.linear1(context)
    # dropout
    output = self.dropout(output)
    # add residual and norm layer
    residual = q
    output = self.layer_norm(residual + output)

    #FNN

    #output2 = output.transpose(1, 2)
    output2 = self.w2(self.relu(self.w1(output)))
    output2 = self.dropout(output2)
    # add residual and norm layer
    output3 = self.layer_norm(output2 + output)

    return output3

def padding_mask(seq_k, seq_q):
  """For masking out the padding part of the keys sequence.
  Args:
    seq_k: Keys tensor, with shape [B, L_k]
    seq_q: Query tensor, with shape [B, L_q]
  Returns:
    A masking tensor, with shape [B, L_1, L_k]
  """
  len_q = seq_q.size(1)
  # `PAD` is 0
  pad_mask = seq_k.eq(0)
  pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
  return pad_mask