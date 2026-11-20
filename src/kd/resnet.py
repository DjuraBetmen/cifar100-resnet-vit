import torch.nn as nn
from torch import Tensor

class Residual(nn.Module):
  def __init__(self, main: nn.Module, shortcut: nn.Module | None = None, act: nn.Module | None = None):
    super().__init__()
    self.main = main
    self.shortcut = shortcut or nn.Identity()
    self.act = act or nn.GELU()

  def forward(self, x: Tensor) -> Tensor:
    return self.act(self.main(x) + self.shortcut(x))

def resBlock(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
  # (B, in, H, W) -> (B, out, H1(H/stride), W1(W/stride)) -> (B, out, H1, W1)
  main = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(),
    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels)
  )
  if stride != 1 or in_channels != out_channels:
    shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
    return Residual(main, shortcut)
  else:
    return Residual(main)

class Resnet(nn.Module):
  def __init__(self, in_channels: int, num_classes: int, width: int = 64, blocks: tuple[int, ...] = (2,2,2,2)):
    super().__init__()

    seq = [
      nn.Conv2d(in_channels, width, kernel_size=(3,3), stride=1, padding=1, bias=False), 
      nn.BatchNorm2d(width), 
      nn.GELU()
    ]

    in_c = width
    for j, n in enumerate(blocks):
      out_c = width * 2**j
      stride = 2 if j > 0 else 1
      seq.append(resBlock(in_c, out_c, stride))
      seq += [resBlock(out_c, out_c, 1) for _ in range(n - 1)]
      in_c = out_c

    seq += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_c, num_classes)]
    self.seq = nn.Sequential(*seq)

  def forward(self, x: Tensor) -> Tensor:
    return self.seq(x)
