# GENERISANO iz src/kd/resnet.py, vit.py, c100.py - ne menjaj rucno, pokreni build_kaggle.py

# ==================== resnet.py ====================
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

# ==================== vit.py ====================
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

# ==================== c100.py ====================
"""CIFAR-100: ResNet-18 ucitelj -> ResNet studenti i ViT student (CE / soft KD / DeiT hard KD).

Koriste se TVOJE klase: kd.resnet.Resnet i kd.vit.ViT.
Na Kaggle ide jedan spojen fajl (build_kaggle.py ga pravi iz resnet.py + vit.py + ovog fajla).
"""
import glob, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader

NC = 100  # broj klasa (CIFAR-100)
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
dev = "cuda" if torch.cuda.is_available() else "cpu"

EP_T = 40                 # epohe ucitelja
EP_S = 60                 # epohe svih studenata (isto za sve, radi poredjenja)
RESNET_WIDTHS = (16, 32)  # ResNet studenti

# ------------------------------------------------------------------ podaci
def find_root() -> str:
  """Ako je dodat Kaggle dataset sa cifar-100-python, koristi njega (bez skidanja); inace 'data'."""
  hits = glob.glob("/kaggle/input/**/cifar-100-python", recursive=True)
  if hits:
    root = hits[0].rsplit("/", 1)[0]
    print("CIFAR-100 iz dataseta:", root)
    return root
  print("CIFAR-100 dataset nije nadjen, skidam (sporo)...")
  return "data"

def loaders(bs: int = 128, root: str | None = None) -> tuple[DataLoader, DataLoader]:
  """Batch: x (B,3,32,32) float32, y (B,) int64 u 0..99."""
  root = root or find_root()
  train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(MEAN, STD)])
  test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
  tr = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
  te = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=test_tf)
  return (DataLoader(tr, bs, shuffle=True, num_workers=4, persistent_workers=True),
          DataLoader(te, 256, num_workers=4, persistent_workers=True))

# ------------------------------------------------------------------ gubici
def kd_loss(s: Tensor, t: Tensor, y: Tensor, T: float = 4.0, alpha: float = 0.9) -> Tensor:
  """Hinton soft KD: alpha * T^2 * KL(teacher || student) + (1-alpha) * CE(student, y). s,t: (B,NC)."""
  soft = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T ** 2
  return alpha * soft + (1 - alpha) * F.cross_entropy(s, y)

def deit_loss(out: tuple[Tensor, Tensor], t: Tensor, y: Tensor) -> Tensor:
  """DeiT hard distillation: 1/2 CE(cls glava, y) + 1/2 CE(dst glava, argmax ucitelja)."""
  cls, dst = out
  return 0.5 * F.cross_entropy(cls, y) + 0.5 * F.cross_entropy(dst, t.argmax(1))

def deit_soft_loss(out: tuple[Tensor, Tensor], t: Tensor, y: Tensor, T: float = 4.0) -> Tensor:
  """DeiT soft distillation: 1/2 CE(cls glava, y) + 1/2 T^2 * KL(ucitelj || dst glava)."""
  cls, dst = out
  soft = F.kl_div(F.log_softmax(dst / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T ** 2
  return 0.5 * F.cross_entropy(cls, y) + 0.5 * soft

def compute_loss(mode: str, out, y: Tensor, t: Tensor | None = None, T: float = 4.0, alpha: float = 0.9) -> Tensor:
  if mode == "ce":   return F.cross_entropy(out, y)
  if mode == "soft": return kd_loss(out, t, y, T, alpha)
  if mode == "deit": return deit_loss(out, t, y)
  if mode == "deit_soft": return deit_soft_loss(out, t, y, T)
  raise ValueError(mode)

# ------------------------------------------------------------------ trening
@torch.no_grad()
def evaluate(model: nn.Module, test: DataLoader) -> float:
  model.eval()
  ok = sum((model(x.to(dev)).argmax(1) == y.to(dev)).sum().item() for x, y in test)
  return ok / len(test.dataset)

def make_optimizer(model: nn.Module, optim: str, lr: float, wd: float) -> torch.optim.Optimizer:
  if optim == "sgd":
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
  # AdamW: bez weight decay-a na bias, LayerNorm, cls/dst tokene i pozicije
  decay, no_decay = [], []
  for n, p in model.named_parameters():
    (no_decay if p.ndim < 2 or n in ("cls", "dst", "positions") else decay).append(p)
  return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                            {"params": no_decay, "weight_decay": 0.0}], lr=lr)

def run(model: nn.Module, epochs: int, train: DataLoader, test: DataLoader,
        tch: nn.Module | None = None, mode: str = "ce", T: float = 4.0, alpha: float = 0.9,
        optim: str = "sgd", lr: float = 0.1, wd: float = 5e-4, warmup: int = 0) -> list[float]:
  """Trenira model, vraca listu test tacnosti po epohama.
  mode: 'ce' (bez ucitelja) | 'soft' (Hinton KD) | 'deit' (hard KD, model mora imati distill=True)."""
  assert (tch is None) == (mode == "ce")
  model.to(dev)
  opt = make_optimizer(model, optim, lr, wd)
  total, wu = epochs * len(train), warmup * len(train)
  sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / max(total - wu, 1))))
  hist: list[float] = []
  for ep in range(epochs):
    model.train()
    for x, y in train:                     # x (B,3,32,32), y (B,)
      x, y = x.to(dev), y.to(dev)
      out = model(x)                       # (B,NC), a za DeiT u treningu par (B,NC),(B,NC)
      t = None
      if tch is not None:
        with torch.no_grad():
          t = tch(x)                       # (B,NC) logiti ucitelja
      loss = compute_loss(mode, out, y, t, T, alpha)
      opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    hist.append(evaluate(model, test))
    print(f"ep {ep} acc {hist[-1]:.4f}", flush=True)
  return hist

# ------------------------------------------------------------------ eksperimenti
def experiments() -> list[tuple[str, object, str, dict]]:
  """(tag, konstruktor, mode, optimizer kwargs)"""
  cnn = dict(optim="sgd", lr=0.1, wd=5e-4, warmup=0)
  vit = dict(optim="adamw", lr=1e-3, wd=0.05, warmup=5)
  exps = []
  for w in RESNET_WIDTHS:
    for mode in ("ce", "soft"):
      exps.append((f"resnet_w{w}_{mode}", lambda w=w: Resnet(3, NC, w), mode, cnn))
  for tag, distill, mode in (("vit_ce", False, "ce"), ("vit_soft", False, "soft"), ("vit_deit", True, "deit")):
    exps.append((tag, lambda d=distill: ViT(3, NC, distill=d), mode, vit))
  return exps

def n_params(m: nn.Module) -> int:
  return sum(p.numel() for p in m.parameters())

def smoke_test(train: DataLoader) -> None:
  """Jedan korak za svaki eksperiment sa nenatreniranim ucitelj-em, da bug iskoci odmah (ne posle 40 min)."""
  x, y = next(iter(train))
  x, y = x.to(dev), y.to(dev)
  tch = Resnet(3, NC).to(dev).eval()
  with torch.no_grad():
    t = tch(x)
  for tag, mk, mode, _ in experiments():
    m = mk().to(dev).train()
    compute_loss(mode, m(x), y, None if mode == "ce" else t).backward()
    assert all(p.grad is not None for p in m.parameters()), f"{tag}: parametar bez gradijenta"
    m.eval()
    assert m(x).shape == (len(x), NC), tag
    print(f"smoke ok: {tag:16s} {n_params(m):>10,} parametara", flush=True)

def main() -> None:
  assert torch.cuda.is_available(), "Ukljuci GPU: Settings -> Accelerator"
  torch.manual_seed(0)
  train, test = loaders()
  results: dict = {}
  def save(): json.dump(results, open("results.json", "w"), indent=1)

  smoke_test(train)

  print("\n=== teacher (Resnet-18, CIFAR-100) ===", flush=True)
  tm = Resnet(3, NC, 64, (2, 2, 2, 2))
  t0 = time.time()
  hist = run(tm, EP_T, train, test)
  tm.eval()
  for p in tm.parameters(): p.requires_grad = False
  torch.save(tm.state_dict(), "teacher100.pt")
  results["teacher"] = {"acc": hist[-1], "params": n_params(tm), "sec": round(time.time() - t0), "hist": hist}; save()
  print(f"teacher: acc {hist[-1]:.4f}", flush=True)

  for tag, mk, mode, kw in experiments():
    torch.manual_seed(0)
    m = mk()
    t0 = time.time()
    print(f"\n=== {tag}  mode={mode}  params {n_params(m):,} ===", flush=True)
    hist = run(m, EP_S, train, test, tch=None if mode == "ce" else tm, mode=mode, **kw)
    torch.save(m.state_dict(), f"{tag}.pt")
    results[tag] = {"acc": hist[-1], "params": n_params(m), "sec": round(time.time() - t0), "hist": hist}; save()
    print(f"{tag}: acc {hist[-1]:.4f} ({results[tag]['sec']} s)", flush=True)

  print("\nREZIME (test tacnost, poslednja epoha)")
  print(f"{'model':>12} {'params':>10} {'ce':>8} {'soft':>8} {'deit':>8}")
  fams: dict[str, dict] = {}
  for tag in results:
    if tag == "teacher": continue
    fam, mode = tag.rsplit("_", 1)
    fams.setdefault(fam, {})[mode] = results[tag]
  for fam, d in fams.items():
    cell = lambda k: f"{d[k]['acc']:.4f}" if k in d else "-"
    print(f"{fam:>12} {next(iter(d.values()))['params']:>10,} {cell('ce'):>8} {cell('soft'):>8} {cell('deit'):>8}")
  print(f"teacher {results['teacher']['acc']:.4f}")

if __name__ == "__main__":
  main()
