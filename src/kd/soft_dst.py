"""Samo jedan eksperiment: ViT + dst token + SOFT destilacija (CIFAR-100).

Ucitelj se ucita iz Kaggle dataseta (teacher100.pt); ako ga nema, istrenira se ponovo (~36 min).
"""
import glob, json, time
import torch
from kd.resnet import Resnet
from kd.vit import ViT
from kd.c100 import NC, EP_T, EP_S, dev, loaders, run, evaluate, n_params

# reference iz prethodnog runa (isti ucitelj, isti hiperparametri, 60 epoha)
REF = {"vit_ce": 0.5561, "vit_soft": 0.6381, "vit_deit (hard+token)": 0.5664}

def get_teacher(train, test) -> Resnet:
  tm = Resnet(3, NC, 64, (2, 2, 2, 2))
  hits = glob.glob("/kaggle/input/**/teacher100.pt", recursive=True)
  if hits:
    tm.load_state_dict(torch.load(hits[0], map_location="cpu"))
    tm.to(dev)
    print("ucitelj ucitan iz", hits[0], flush=True)
  else:
    print("teacher100.pt nije nadjen, treniram ucitelja (~36 min)", flush=True)
    run(tm, EP_T, train, test)
  tm.eval()
  for p in tm.parameters(): p.requires_grad = False
  print(f"ucitelj test acc: {evaluate(tm, test):.4f}", flush=True)
  return tm

@torch.no_grad()
def heads_acc(model: ViT, test) -> tuple[float, float]:
  """Tacnost svake glave posebno (u train modu ViT vraca par glava; nema dropout-a ni BN-a)."""
  model.train()
  ok_c = ok_d = 0
  for x, y in test:
    c, d = model(x.to(dev))
    y = y.to(dev)
    ok_c += (c.argmax(1) == y).sum().item()
    ok_d += (d.argmax(1) == y).sum().item()
  model.eval()
  n = len(test.dataset)
  return ok_c / n, ok_d / n

def main() -> None:
  assert torch.cuda.is_available(), "Ukljuci GPU: Settings -> Accelerator"
  torch.manual_seed(0)
  train, test = loaders()
  tm = get_teacher(train, test)

  torch.manual_seed(0)
  m = ViT(3, NC, distill=True)
  print(f"\n=== vit_deit_soft  params {n_params(m):,} ===", flush=True)
  t0 = time.time()
  hist = run(m, EP_S, train, test, tch=tm, mode="deit_soft", optim="adamw", lr=1e-3, wd=0.05, warmup=5)
  torch.save(m.state_dict(), "vit_deit_soft.pt")
  c, d = heads_acc(m, test)
  json.dump({"acc": hist[-1], "cls_head": c, "dst_head": d, "params": n_params(m),
             "sec": round(time.time() - t0), "hist": hist}, open("results_soft_dst.json", "w"), indent=1)

  print("\nREZIME (CIFAR-100, test tacnost, poslednja epoha)")
  for k, v in REF.items():
    print(f"{k:>24} {v:.4f}")
  print(f"{'vit_deit_soft (NOVO)':>24} {hist[-1]:.4f}   (cls glava {c:.4f}, dst glava {d:.4f})")

if __name__ == "__main__":
  main()
