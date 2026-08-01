# TEDFusion
Text-Driven Fusion for Infrared and Visible Images: Achieving Image Scene Adaptation on Hyperbolic Space
====
Accetped by [ICML 2026] 🔗"https://arxiv.org/pdf/2606.15104"


⬇️Model Download
----
The model can be found in https://pan.baidu.com/s/1Daz11onyLza_jLuXcBy1mA?pwd=h5b6 and the password is: h5b6



⚙️Training
----
```bash
python train_fusion.py `
  --dataset_path "./dataset/train_MSRS" `
  --samples_per_epoch 800 `
  --epochs 15 `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --grad_clip_norm 1.0
```


✔️Testing
----
For inputs with standard image dimensions, please use the following command for testing:
```bash
python test_from_dataset.py  --weights_path "pretrained_weights/train/TEDFusion.pth" --dataset_path "./dataset/test_LLVIP" --save_path "./results/LLVIP"
```
