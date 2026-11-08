# Arabic Calligraphy Script Classification

> Can a model tell **Naskh** from **Thuluth** from **Diwani**? Given an image of Arabic calligraphy, this project predicts which of **12 scripts** it is written in.

🏆 **5th of 21 teams** · Olo Academy Kaggle competition (AIEP bootcamp, Olo Academy × MCIT)
📈 **Macro F1: 0.08 → 0.956** (provided baseline → final 5-fold CV)

![Macro F1 by version](results.png)

## The challenge
- Only **800 training images** (~67 per script) across 12 classes
- Scored by **macro F1**, so every script counts equally
- No external calligraphy data, free Kaggle GPU only

## What worked
| Version | Key change | Macro F1 |
|---|---|---|
| Baseline | 1 conv layer, trained from scratch | 0.08 |
| v1 | ImageNet-pretrained **EfficientNet-B0** + augmentation | 0.76 (0.77 public LB) |
| v2 | **5-fold CV** ensemble of EfficientNet-B0 + **ConvNeXt-Tiny**, test-time augmentation, class balancing | 0.93 |
| v3 | **ConvNeXt-Tiny + EfficientNetV2-S** at 384 px | **0.956** |

**Design choices**
- **Transfer learning:** with ~67 images per class, pretrained weights gave the biggest single jump.
- **No horizontal flips:** Arabic has a writing direction, so flipping creates unrealistic samples.
- **Class balancing:** keeps predictions spread across all 12 scripts, which directly helps macro F1.

*v3 models were selected using the same CV folds, so 0.956 is slightly optimistic.*

## Run it
1. Open `arabic_calligraphy_classification.ipynb` on Kaggle and attach the competition data (GPU on).
2. Run all cells; each version writes its own submission.

**Stack:** Python · PyTorch · torchvision · scikit-learn · pandas · Matplotlib

Competition data is not included. Team of 2; I wrote all of the code.

**Sarah Al-Bahesh** · [LinkedIn](https://www.linkedin.com/in/sarah-albahesh) · [GitHub](https://github.com/saravii7)
