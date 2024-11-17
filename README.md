---
pipeline_tag: image-to-image
tags:
- HiT-SR
- image super-resolution
- transformer
- efficient transformer
---

<h1>
 HiT-SR: Hierarchical Transformer for Efficient Image Super-Resolution
 </h1>

<h3><a href="https://github.com/XiangZ-0/HiT-SR">[Github]</a> | <a href="https://huggingface.co/papers/2407.05878">[Paper]</a> | <a href="https://1drv.ms/b/c/de821e161e64ce08/EYmRy-QOjPdFsMRT_ElKQqABYzoIIfDtkt9hofZ5YY_GjQ?e=2Iapqf">[Supp]</a> | <a href="https://www.youtube.com/watch?v=9rO0pjmmjZg">[Video]</a> | <a href="https://1drv.ms/f/c/de821e161e64ce08/EuE6xW-sN-hFgkIa6J-Y8gkB9b4vDQZQ01r1ZP1lmzM0vQ?e=aIRfCQ">[Visual Results]</a> </h3>
<div></div>

HiT-SR is a general strategy to improve transformer-based SR methods. We apply our HiT-SR approach to improve [SwinIR-Light](https://github.com/JingyunLiang/SwinIR), [SwinIR-NG](https://github.com/rami0205/NGramSwin) and [SRFormer-Light](https://github.com/HVision-NKU/SRFormer), corresponding to our HiT-SIR, HiT-SNG, and HiT-SRF. Compared with the original structure, our improved models achieve better SR performance while reducing computational burdens.


## 🛠️ Setup
Install the dependencies under the working directory:
```
git clone https://huggingface.co/XiangZ/hit-sr
cd hit-sr
pip install -r requirements.txt
```

## 🚀 Usage
For each HiT-SR model, we provide 2x, 3x, 4x upscaling versions:
| Repo Name         |   | Model   |   | Upscale |
|-------------------|---|---------|---|---------|
| `XiangZ/hit-sir-2x` |   | HiT-SIR |   | 2x      |
| `XiangZ/hit-sir-3x` |   | HiT-SIR |   | 3x      |
| `XiangZ/hit-sir-4x` |   | HiT-SIR |   | 4x      |
| `XiangZ/hit-sng-2x` |   | HiT-SNG |   | 2x      |
| `XiangZ/hit-sng-3x` |   | HiT-SNG |   | 3x      |
| `XiangZ/hit-sng-4x` |   | HiT-SNG |   | 4x      |
| `XiangZ/hit-srf-2x` |   | HiT-SRF |   | 2x      |
| `XiangZ/hit-srf-3x` |   | HiT-SRF |   | 3x      |
| `XiangZ/hit-srf-4x` |   | HiT-SRF |   | 4x      |

To test the model (use hit-srf-4x as an example):
```
from hit_sir_arch import HiT_SIR
from hit_sng_arch import HiT_SNG
from hit_srf_arch import HiT_SRF
import cv2

# use GPU (True) or CPU (False)
cuda_flag = True

# initialize model (change model and upscale according to your setting)
model = HiT_SRF(upscale=4) 

# load model (change repo_name according to your setting)
repo_name = "XiangZ/hit-srf-4x"
model = model.from_pretrained(repo_name)
if cuda_flag:
    model.cuda()

# test and save results
sr_results = model.infer_image("path-to-input-image", cuda=cuda_flag)
cv2.imwrite("path-to-output-location", sr_results)
```

## 📎 Citation

If you find the code helpful in your research or work, please consider citing the following paper.

```
@inproceedings{zhang2024hitsr,
    title={HiT-SR: Hierarchical Transformer for Efficient Image Super-Resolution},
    author={Zhang, Xiang and Zhang, Yulun and Yu, Fisher},
    booktitle={ECCV},
    year={2024}
}
```
