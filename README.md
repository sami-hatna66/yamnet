# PyTorch YAMNet

This repository implements the YAMNet model for audio classification in PyTorch. YAMNet was originally released in TensorFlow by Google. This implementation is adapted from [Torch AudioSet](https://github.com/w-hc/torch_audioset), which only supports inference using pretrained weights. In contrast, this version adds full support for training from scratch. It also adds support for an enhanced version of YAMNet which replaces the MobileNetV1 backbone with MobileNetV3.

![architecture](https://github.com/user-attachments/assets/03ab628b-8bdc-4574-b92d-0a61683805a2)

## Usage

[example.py](example.py) contains example code training YAMNet on the ESC50 dataset. The project contains dataloaders and scripts for downloading the ESC50 and FSD50K datasets for audio classification. Training YAMNet on ESC50 is as simple as the following:

```bash
pip install -r requirements.txt
./download_esc50.sh
python3 example.py ./ESC-50-master ./log.txt ./ckpt.pt
```
