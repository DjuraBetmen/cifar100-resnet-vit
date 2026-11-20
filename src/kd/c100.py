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
from kd.resnet import Resnet
from kd.vit import ViT

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
