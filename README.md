# TEDFusion
Text-Driven Fusion for Infrared and Visible Images: Achieving Image Scene Adaptation on Hyperbolic Space
====
Accetped by [ICML 2026] 🔗"https://arxiv.org/pdf/2606.15104"


⬇️Model Download
----
The model can be found in https://pan.baidu.com/s/1Daz11onyLza_jLuXcBy1mA?pwd=h5b6 and the password is: h5b6

Artifact-resistant MSRS fine-tuning
----
Fine-tune from the released checkpoint with:

```bash
python train_fusion.py --weights /path/to/checkpoint.pth
```

When `--weights` is supplied, training uses a conservative learning rate
(`2e-5`), low weight decay (`1e-4`), gradient clipping, and keeps the pretrained
PixelShuffle convolutions frozen. These settings do not change the network or
the published losses, and add negligible training time. Use
`--train_upsamplers` only for an ablation that intentionally updates those
phase-sensitive filters. Explicit `--lr`, `--weight_decay`, and
`--grad_clip_norm` values override the defaults.

For full MSRS, use one dataset argument for all task slots:

```bash
python train_fusion.py --dataset_path ./dataset/train_MSRS --weights /path/to/checkpoint.pth
```

Training draws 800 random pairs per epoch by default, matching the original
small-MSRS run. Identical task roots are deduplicated, so expanding MSRS no
longer changes an epoch from 800 to 8664 samples. Use
`--samples_per_epoch 0` only when intentionally drawing once per unique pair.
