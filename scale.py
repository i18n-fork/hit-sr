#!/usr/bin/env python

from tqdm import tqdm
import os
import cv2
from os.path import dirname, join, abspath
from fire import Fire


@Fire
def main(input_dir, output_dir, scale=4):
  root = dirname(dirname(abspath(__file__)))
  input_dir = join(root, input_dir)
  output_dir = join(root, output_dir)
  os.makedirs(output_dir, exist_ok=True)
  # initialize model (change model and upscale according to your setting)

  if scale == 4:
    from hit_srf_arch import HiT_SRF

    repo_name = "XiangZ/hit-srf-4x"
    model = HiT_SRF(upscale=4)
  else:
    from hit_sir_arch import HiT_SIR

    repo_name = "XiangZ/hit-sir-2x"
    model = HiT_SIR(upscale=2)

  print("\n%s\n", repo_name)

  cuda_flag = False

  model = model.from_pretrained(repo_name)

  # load model (change repo_name according to your setting)
  # if cuda_flag:
  #     model.cuda()

  li = []
  for root, dirs, files in os.walk(input_dir):
    for file in files:
      if file.endswith(".png"):
        file_path = os.path.join(root, file)
        li.append((file, file_path))

  for file, file_path in tqdm(li):
    tqdm.write(file)
    sr_results = model.infer_image(file_path, cuda=cuda_flag)
    os.remove(file_path)
    cv2.imwrite(join(output_dir, file), sr_results)
  return
