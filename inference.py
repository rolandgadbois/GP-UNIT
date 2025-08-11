import os
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from model.generator import Generator
from model.content_encoder import ContentEncoder
from util import load_image, save_image
import glob

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Inference of GP-UNIT")
        group = self.parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--content", type=str, help="Path to the content image")
        group.add_argument("--content_dir", type=str, help="Path to a folder of content images")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="Path to save the output images")
        self.parser.add_argument("--name", type=str, default='translation', help="Filename prefix for the output")
        self.parser.add_argument("--generator_path", type=str, default='./checkpoint/generator.pt', help="Path to the saved generator")
        self.parser.add_argument("--content_encoder_path", type=str, default='./checkpoint/content_encoder.pt', help="Path to the saved content encoder")
        self.parser.add_argument("--device", type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'")
        self.parser.add_argument("--save_overview", action='store_true', help="Save side-by-side overview image with original and generated")
        
    def parse(self):
        self.opt = self.parser.parse_args()        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print(f'{name}: {value}')
        return self.opt

if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    device = args.device
    netEC = ContentEncoder()
    netEC.eval()
    netG = Generator()
    netG.eval()
    
    netEC.load_state_dict(torch.load(args.content_encoder_path, map_location=lambda storage, loc: storage))
    ckpt = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
    netG.load_state_dict(ckpt['g_ema'])
    
    netEC = netEC.to(device)
    netG = netG.to(device)
    
    print('Load models successfully!')

    # Prepare list of images
    if args.content_dir:
        image_paths = []
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.content_dir, ext)))
        image_paths = sorted(image_paths)
        print(f'Found {len(image_paths)} images in {args.content_dir}')
    else:
        image_paths = [args.content]
    
    os.makedirs(args.output_path, exist_ok=True)

    for img_path in image_paths:
      save_name = args.name + '_' + os.path.basename(img_path).split('.')[0]
      with torch.no_grad():
          Ix = F.interpolate(load_image(img_path), size=256, mode='bilinear', align_corners=True)
          content_feature = netEC(Ix.to(device), get_feature=True)
          I_yhat, _ = netG(content_feature)

      save_image(I_yhat[0].cpu(), os.path.join(args.output_path, save_name + '.jpg'))
      
      if args.save_overview:
          save_image(torchvision.utils.make_grid(torch.cat([Ix, I_yhat.cpu()], dim=0), 2, 1), 
                    os.path.join(args.output_path, save_name + '_overview.jpg'))
      
      print(f'Processed and saved outputs for {img_path}')
    
    print('All images processed successfully!')
