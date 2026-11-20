import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F

class Attention(nn.Module):
  def __init__(self, dim: int, heads: int):
    super().__init__()
    assert dim % heads == 0
    self.h, self.dh = heads, dim // heads
    self.qkv = nn.Linear(dim, 3 * dim)
    self.proj = nn.Linear(dim, dim)

  def forward(self, x: Tensor) -> Tensor:
    B, N, D = x.shape
    # (B,N,D) -> (B,N,3*D) -> (B,N,h,3*Dh) -> (B,h,N,3*Dh) -> 3 (B,h,N,Dh)
    q, k, v = self.qkv(x).reshape(B, N, self.h, -1).transpose(1, 2).chunk(3, -1)
    # (B,h,N,Dh) @ (B,h,Dh,N) -> (B,h,N,N) @ (B,h,N,Dh) -> (B,h,N,Dh) -> (B,N,h,Dh) -> (B,N,D)
    return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))

class vitBlock(nn.Module):
  def __init__(self, dim: int = 192, heads: int = 3, mlp_ratio: int = 4):
    super().__init__()
    self.ln1 = nn.LayerNorm(dim)
    self.msa = Attention(dim, heads)
    self.ln2 = nn.LayerNorm(dim)
    self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))

  def forward(self, x: Tensor) -> Tensor:
    # MSA block
    x = x + self.msa(self.ln1(x))
    # MLP block: (B,N,D) -> (B,N,r*D) -> (B,N,D)
    return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
  def __init__(self, in_channels: int, num_classes: int, img_size: int = 32, patch: int = 4,
               dim: int = 192, depth: int= 6, heads: int = 3, mlp_ratio: int = 4, distill: bool = False):
    super().__init__()
    assert img_size % patch == 0
    self.distill = distill
    self.embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
    self.cls = nn.Parameter(torch.zeros(1, 1, dim))
    self.dst = nn.Parameter(torch.zeros(1, 1, dim)) if distill else None
    self.positions = nn.Parameter(torch.randn(1, (img_size // patch) ** 2 + (2 if distill else 1), dim) * 0.02)
    self.blocks = nn.Sequential(*[vitBlock(dim, heads, mlp_ratio) for _ in range(depth)])
    self.norm = nn.LayerNorm(dim)
    self.cls_head = nn.Linear(dim, num_classes)
    self.dst_head = nn.Linear(dim, num_classes) if distill else None
    
  def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
    # (B,in,32,32) -> (B,D,(32-4)/4 + 1,(32-4)/4 + 1) -> (B,D,8,8) -> (B,D,64) -> (B,64,D)
    x = self.embed(x).flatten(2).transpose(1, 2)
    # (1,1,D) -> (B,1,D)
    cls_tokens = self.cls.expand(len(x), -1, -1)
    if self.distill:
      # (1,1,D) -> (B,1,D)
      dst_tokens = self.dst.expand(len(x), -1, -1)
      # (B,1,D),(B,1,D),(B,64,D) -> (B,66,D) + (1,66,D) -> (B,66,D)
      x = torch.cat((cls_tokens, dst_tokens, x), dim=1) + self.positions
    else:
      # (B,1,D),(B,64,D) -> (B,65,D) + (1,65,D) -> (B,65,D)
      x = torch.cat((cls_tokens, x), dim=1) + self.positions
   
    # (B,N,D) -> Transformer Block -> (B,N,D)
    x = self.norm(self.blocks(x))
    if not self.distill:
      # (B,D) -> (B,num_classes)
      return self.cls_head(x[:, 0])
    elif self.training:
      # (B,D),(B,D) -> (B,num_classes), (B,num_classes)
      return (self.cls_head(x[:, 0]),  self.dst_head(x[:, 1]))
    else:
      # (B,D),(B,D) -> (B,num_classes)
      return (self.cls_head(x[:, 0]) + self.dst_head(x[:, 1])) / 2
