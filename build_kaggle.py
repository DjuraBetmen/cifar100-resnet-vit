"""Spaja src/kd/*.py u jedan fajl koji se lepi u jednu Kaggle celiju.

Pokretanje (iz korena projekta):  uv run python build_kaggle.py
Pravi:
  kaggle_c100.py      - ceo eksperiment (ucitelj + 7 studenata)
  kaggle_soft_dst.py  - samo ViT + dst token + soft KD (ucitelj iz dataseta)
Posle svake izmene u src/kd/*.py ponovo pokreni ovo.
"""
import re
from pathlib import Path

SRC = Path("src/kd")
MAIN_BLOCK = re.compile(r'^if __name__ == "__main__":\n  main\(\)\n', flags=re.M)

def build(parts: list[str], out_name: str) -> None:
  out = ['# GENERISANO iz src/kd/{} - ne menjaj rucno, pokreni build_kaggle.py\n'.format(", ".join(parts))]
  for i, name in enumerate(parts):
    text = (SRC / name).read_text(encoding="utf-8")
    text = re.sub(r"^from kd\.\w+ import .*\n", "", text, flags=re.M)   # paketni importi ne postoje u spojenom fajlu
    if i < len(parts) - 1:
      text = MAIN_BLOCK.sub("", text)                                   # main() se pokrece samo iz poslednjeg dela
    out.append(f"\n# {'=' * 20} {name} {'=' * 20}\n{text}")
  Path(out_name).write_text("".join(out), encoding="utf-8")
  print("napisan", out_name)

build(["resnet.py", "vit.py", "c100.py"], "kaggle_c100.py")
build(["resnet.py", "vit.py", "c100.py", "soft_dst.py"], "kaggle_soft_dst.py")
