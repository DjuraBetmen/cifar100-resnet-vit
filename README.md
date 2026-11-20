# kd: destilacija znanja (ResNet → ResNet / ViT / DeiT) na CIFAR-100

Prvi korak ka razumevanju BEVFormer-a: od nule sam napisao ResNet i ViT u PyTorch-u, pa na njima
ispitao Hintonovu destilaciju znanja i DeiT destilacioni token. Cilj nije rekord, nego da se
razume šta se desi kad se isti ViT trenira sa i bez učitelja.

Plan koji ovo otvara: KD → ResNet → ViT → DeiT → DETR → Deformable DETR → geometrija kamere / BEV (LSS) → BEVFormer.

## Šta je u repozitorijumu

| Fajl | Sadržaj |
|---|---|
| `src/kd/resnet.py` | ResNet po blokovima: `Residual`, `resBlock`, `Resnet(in, classes, width, blocks)` |
| `src/kd/vit.py` | ViT (`Attention`, `vitBlock`, `ViT`) sa opcionim destilacionim tokenom (`distill=True`) |
| `src/kd/c100.py` | podaci, loss-ovi, petlja za trening, definicije eksperimenata |
| `src/kd/soft_dst.py` | dodatni eksperiment: ViT + dst token + soft destilacija |
| `build_kaggle.py` | spaja module u jedan fajl za jednu Kaggle ćeliju |
| `kaggle_c100.py`, `kaggle_soft_dst.py` | generisani fajlovi (ne menjaj ručno) |
| `results/*.json` | tačnost po epohama, broj parametara i vreme za svaki trening |

## Modeli

**ResNet.** Stem je konvolucija 3×3 stride 1 bez max-poola (slike su 32×32), pa četiri faze
sa 2 bloka (`blocks=(2,2,2,2)` je ResNet-18, druge torke daju druge dubine). Širina faze je
`width·2^k`. Blok je `act(main(x) + shortcut(x))`; shortcut je 1×1 konvolucija + BN kad se
menja stride ili broj kanala, inače identitet. Aktivacija je GELU. Glava: AdaptiveAvgPool → Flatten → Linear.
Učitelj je `Resnet(width=64)` sa 11.2M parametara; studenti su širine 16 (0.7M) i 32 (2.8M).

**ViT.** Patch 4×4 (konvolucija kernel = stride = 4), 64 tokena + `[CLS]`, naučene pozicije,
6 pre-norm blokova, `dim=192`, 3 glave, MLP faktor 4; 2.71M parametara. Klasifikacija iz `[CLS]` tokena.
Sa `distill=True` dodaje se `[DST]` token i druga glava; u treningu model vraća par glava,
u evaluaciji prosek obe (kao u DeiT radu).

## Destilacija (loss-ovi)

- **CE**: običan trening na labelama.
- **Soft (Hinton)**: `α·T²·KL(učitelj ‖ student) + (1−α)·CE`, T=4, α=0.9.
- **Hard (DeiT)**: `½·CE(cls, y) + ½·CE(dst, argmax učitelja)`.
- **Soft + dst token**: `½·CE(cls, y) + ½·T²·KL(učitelj ‖ dst)`, T=4.

## Podešavanje treninga

- Podaci: CIFAR-100, augmentacija samo RandomCrop(32, padding 4) + horizontalni flip (bez Mixup/CutMix).
- ResNet-ovi: SGD (Nesterov, momentum 0.9), lr 0.1, wd 5e-4, cosine; **učitelj 40 epoha, studenti 60**.
- ViT: AdamW, lr 1e-3, wd 0.05 (bez wd na bias, LayerNorm, tokene i pozicije), warmup 5 epoha + cosine, 60 epoha.
- Batch 128, seed 0, **jedan run po konfiguraciji**. GPU na Kaggle-u (ceo eksperiment ≈ 4.4 h, dodatni oko 46 min).

## Rezultati (test tačnost na CIFAR-100, poslednja epoha)

| Model | Parametri | CE | Soft KD | Hard KD + dst token | Soft KD + dst token |
|---|---|---|---|---|---|
| Učitelj ResNet-18 (w64) | 11.2M | 75.97 | | | |
| ResNet w32 | 2.8M | 74.37 | 76.46 | | |
| ResNet w16 | 0.7M | 69.57 | 72.83 | | |
| ViT | 2.7M | 55.61 | **63.81** | 56.64 | 62.43 |

Za varijantu soft + dst token: cls glava 61.30, dst glava 63.37, prosek 62.43 (`results/results_soft_dst.json`).

## Šta smo zaključili

1. **ViT iz nule na 50k slika je daleko iza ResNet-a iste veličine** (55.6 vs 74.4). Nema induktivne pristrasnosti za lokalnost, a podataka je malo. Krive pokazuju da ViT-CE stagnira posle ~30. epohe.
2. **Soft KD pomaže najviše slabijem modelu**: +3.3 (ResNet w16), +2.1 (w32), +8.2 (ViT). Učitelj je konvolucioni, pa meke labele prenose i deo onoga što ViT sam ne nauči. Iz istog razloga KD ubrzava konvergenciju.
3. **Hard KD sa dst tokenom jedva pomaže (+1.0)**, kriva skoro prati CE. Moje objašnjenje (nije provereno): učitelj je na trening slikama skoro uvek u pravu (slaba augmentacija), pa je `argmax(učitelj) ≈ y` i dst glava ne dobija novu informaciju. DeiT koristi jake augmentacije, pa mu hard target nosi više.
4. **Dst token sam nije bitan**: soft + token (62.4) nije bolji od soft bez tokena (63.8); dst glava sama (63.4) uči skoro isto što i običan soft. Cls glava je sa 55.6 skočila na 61.3 samo zato što deli backbone sa dst glavom. Prosek glava je lošiji od same dst glave.
5. KD je skuplji u računu: soft treninzi traju skoro duplo duže jer svaki korak uključuje prolaz kroz učitelja.

## Ograničenja

- Jedan seed po ćeliji; razlike od 1–2 poena su u okviru šuma (ranije viđeno 0.1–0.6 poena na ResNet-ovima, a ViT je verovatno nestabilniji).
- Učitelj ima 40 epoha, a studenti 60, pa tvrdnja "student premašuje učitelja" (ResNet w32 + soft, 76.46 vs 75.97) nije čista.
- Hiperparametri nisu podešavani posebno za svaku varijantu. Soft koristi α=0.9 (90% signala od učitelja), a soft + token 0.5/0.5, pa to poređenje nije potpuno jednako.
- Hipoteza o hard KD (tačka 3) nije direktno testirana. Predložen test: Mixup/CutMix u sve tri ViT varijante.
- Ranije, preliminarno na CIFAR-10 (kod uklonjen iz repozitorijuma): učitelj 93.6, student bez KD 82.9, sa KD 84.8.

## Pokretanje

Trening je namenjen GPU-u (Kaggle); lokalno se mogu samo proveriti oblici i loss-ovi.

```bash
uv sync
uv run python build_kaggle.py     # pravi kaggle_c100.py i kaggle_soft_dst.py
```

Na Kaggle-u: napravi dataset od foldera `cifar-100-python` (da se podaci ne skidaju), nalepi
`kaggle_c100.py` u jednu ćeliju, uključi GPU i pokreni *Save & Run All*. Za dodatni eksperiment
dodaj dataset sa `teacher100.pt` (izlaz prvog runa) i nalepi `kaggle_soft_dst.py`.
Istrenirane težine nisu u repozitorijumu (do ≈ 45 MB po modelu); brojevi i krive su u `results/`.

## Sledeće

DETR → Deformable DETR → kamera i BEV (LSS) → BEVFormer.
