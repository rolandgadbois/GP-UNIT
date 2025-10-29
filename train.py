import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import utils
import numpy as np
import copy
import argparse
from tqdm import tqdm

from model.generator import Generator
from model.discriminator import Discriminator
from model.content_encoder import ContentEncoder
from model.vgg import VGGLoss
from model.arcface.id_loss import IDLoss

from util import load_image, visualize, adv_loss, r1_reg, divide_pred, moving_average
from dff_util import run_dff_pipeline, build_channel_mask, compute_dff_loss
from dataset import create_unpaired_dataloader

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Train Adversarial Image Translation of GP-UNIT")
        self.parser.add_argument("--task", type=str, help="task type, e.g. cat2dog")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--iter", type=int, default=75000, help="iterations")
        self.parser.add_argument("--batch", type=int, default=16, help="batch size")
        self.parser.add_argument("--content_encoder_path", type=str, default='./checkpoint/content_encoder.pt', help="path to the saved content encoder")
        self.parser.add_argument("--identity_path", type=str, default='./checkpoint/model_ir_se50.pth', help="path to the identity model")
        self.parser.add_argument("--lambda_reconstruction", type=float, default=1.0, help="the weight of reconstruction loss")
        self.parser.add_argument("--lambda_content", type=float, default=1.0, help="the weight of content loss")
        self.parser.add_argument("--lambda_mask", type=float, default=1.0, help="the weight of mask loss")
        self.parser.add_argument("--lambda_id", type=float, default=1.0, help="the weight of identity loss")
        self.parser.add_argument("--lambda_dff", type=float, default=1.0, help="weight for DFF perceptual loss")
        self.parser.add_argument("--source_paths", type=str, nargs='+', help="the path to the training images in each source domain")
        self.parser.add_argument("--target_paths", type=str, nargs='+', help="the path to the training images in each target domain")
        self.parser.add_argument("--source_num", type=int, nargs='+', default=[0], help="the number of the training images in each source domain")
        self.parser.add_argument("--target_num", type=int, nargs='+', default=[0], help="the number of the training images in each target domain")
        self.parser.add_argument("--use_allskip", action="store_true", help="use dynamic skip connection to compute Lrec")
        self.parser.add_argument("--use_idloss", action="store_true", help="use identity loss")
        self.parser.add_argument("--save_every", type=int, default=5000, help="interval of saving a checkpoint")
        self.parser.add_argument("--save_begin", type=int, default=50000, help="when to start saving a checkpoint")
        self.parser.add_argument("--visualize_every", type=int, default=1000, help="interval of saving an intermediate result")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path to the saved models")  
        self.parser.add_argument("--dff_dataset", type=str, default=None, help="path to images used for DFF concept extraction")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.source_num[0] == 0:
            self.opt.source_num = [int(1e8)] * len(self.opt.source_paths)
        if self.opt.target_num[0] == 0:
            self.opt.target_num = [int(1e8)] * len(self.opt.target_paths)        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

def train(args, dataloader, netG, netD, optimizer_G, optimizer_D, netG_ema, 
          vgg_loss, id_loss=None, device='cuda', dff_masks = None):
    
    pbar = tqdm(range(args.iter), initial=0, smoothing=0.01, ncols=130, dynamic_ncols=False)
    
    netG.train()
    netD.train()
    netG_ema.eval()
    iterator = iter(dataloader)
    for idx in pbar:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            data = next(iterator) 

        x, y = data['source'], data['target']
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            cfeat_x = netEC(x, get_feature=True)
            cfeat_y = netEC(y, get_feature=True)

        for i, mask in enumerate(dff_masks):
            # Skip if the layer doesn't exist or size mismatch
            if i < len(cfeat_x):
                cfeat_x[i] = cfeat_x[i] * mask.view(1, -1, 1, 1)
                cfeat_y[i] = cfeat_y[i] * mask.view(1, -1, 1, 1)

        loss_dict = {}

        # No flipping of style image anymore, just use y directly
        y_ = y

        # translation
        yhat, masks = netG(cfeat_x)
        # reconstruction
        ybar, _ = netG(cfeat_y, useskip=args.use_allskip)

        fake_and_real = torch.cat([yhat, y], dim=0)
        preds = netD(fake_and_real)
        fake_pred, real_pred = divide_pred(preds)
        Lgadv = adv_loss(fake_pred, 1)

        Lcon = F.l1_loss(netEC(yhat), cfeat_x[0]) * args.lambda_content

        Lmsk = torch.tensor(0.0, device=device)
        for mask in masks:
            Lmsk += torch.mean(mask) * args.lambda_mask

        Lrec = (F.l1_loss(ybar, y) + vgg_loss(ybar, y)) * args.lambda_reconstruction

        Lid = torch.tensor(0.0, device=device) 
        if args.use_idloss: 
            Lid = id_loss(yhat, y) * args.lambda_id

        Ldff = compute_dff_loss(netEC, x, yhat, dff_masks, target_layers=[1, 2, 3, 4], device=device) * args.lambda_dff
        loss_dict['dff'] = Ldff

        loss_dict['g'] = Lgadv
        loss_dict['con'] = Lcon    
        loss_dict['msk'] = Lmsk
        loss_dict['rec'] = Lrec
        if args.use_idloss:
            loss_dict['id'] = Lid

        g_loss = Lgadv + Lcon + Lmsk + Lrec + Lid + Ldff

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        with torch.no_grad():
            yhat, _ = netG(cfeat_x)

        y.requires_grad_()
        fake_and_real = torch.cat([yhat.detach(), y], dim=0)
        preds = netD(fake_and_real)
        fake_pred, real_pred = divide_pred(preds)    

        Ldadv = adv_loss(real_pred, 1) + adv_loss(fake_pred, 0)
        Lr1 = r1_reg(real_pred, y)    

        d_loss = Ldadv + Lr1

        loss_dict['d'] = Ldadv
        loss_dict['r1'] = Lr1  

        optimizer_D.zero_grad()      
        d_loss.backward()
        optimizer_D.step()

        moving_average(netG, netG_ema, beta=0.999)

        message = ''
        for k, v in loss_dict.items():
            v = v.mean().float()
            message += 'L%s: %.3f ' % (k, v)
        pbar.set_description((message))

        if ((idx+1) >= args.save_begin and (idx+1) % args.save_every == 0) or (idx+1) == args.iter:
            torch.save(
                {
                    #"g": netG.state_dict(),
                    #"d": netD.state_dict(),
                    "g_ema": netG_ema.state_dict(),
                    #"g_optim": optimizer_G.state_dict(),
                    #"d_optim": optimizer_D.state_dict(),
                    #"args": args,
                },
                f"%s/%s-%05d.pt"%(args.model_path, args.task, idx+1),
            )

        if idx == 0 or (idx+1) % args.visualize_every == 0 or (idx+1) == args.iter:
            with torch.no_grad():
                yhat2, _ = netG_ema(cfeat_x)

            viznum = min(args.batch, 8)
            sample = F.adaptive_avg_pool2d(torch.cat((x[0:viznum].cpu(), y[0:viznum].cpu(), 
                                                        yhat[0:viznum].cpu(), yhat2[0:viznum].cpu()), dim=0), 128)
            utils.save_image(
                sample,
                f"log/%s/%05d.jpg"%(args.task, (idx+1)),
                nrow=viznum,
                normalize=True,
                #range=(-1, 1),
            )

if __name__ == "__main__":

    parser = TrainOptions()
    args = parser.parse()
    print('*'*98)
    
    if not os.path.exists("log/%s/"%(args.task)):
        os.makedirs("log/%s/"%(args.task))
    
    device = 'cuda'
    netEC = ContentEncoder()
    netEC.load_state_dict(torch.load(args.content_encoder_path, map_location=lambda storage, loc: storage))
    netEC = netEC.to(device)
    for param in netEC.parameters():
        param.requires_grad = False

    print("Running DFF pipeline...")
    target_layers = [1, 2, 3, 4]
    
    if args.dff_dataset is None or not os.path.exists(args.dff_dataset):
        raise ValueError("You must provide a valid path for --dff_dataset")
    
    images, dff_maps, concepts_per_layer = run_dff_pipeline(
        image_dir=args.dff_dataset,
        content_encoder=netEC,
        target_layers=target_layers,
        n_components=10,
        image_size=(256, 256)
    )

    keep_concepts = {
        0: [3, 4, 6, 7, 8, 9],  # Layer 1
        1: [1, 2, 6, 7, 8, 9],  # Layer 2
        2: [1, 4, 5, 6, 7, 8, 9],     # Layer 3
        3: [1, 2, 3, 4, 5, 6, 7, 8],     # Layer 4
    }

    dff_masks = []
    for i, W in enumerate(concepts_per_layer):
        C = W.shape[0]
        mask = build_channel_mask(W, keep_concepts[i], C=C)
        dff_masks.append(mask)  # mask is np.array of shape [C]
    
    dff_masks = [torch.tensor(m).to(device) for m in dff_masks]
    print("DFF masks ready.")
        
    netG = Generator().to(device)
    netG_ema = Generator().to(device)
    netD = Discriminator().to(device)

    netG.init_weights('kaiming', 0.02)
    netD.init_weights('kaiming', 0.02)

    netG_ema = copy.deepcopy(netG)    
    for param in netG_ema.parameters():
        param.requires_grad = False    
        
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.0, 0.99), weight_decay=1e-4)
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.99), weight_decay=1e-4)
    
    print('Create models successfully!')
    
    # for image translation
    dataloader = create_unpaired_dataloader(args.source_paths, args.target_paths, 
                                            args.source_num, args.target_num, args.batch)

    print('Create dataloaders successfully!')
    
    vgg_loss = VGGLoss()
    vgg_loss.vgg = vgg_loss.vgg.to(device)
    if args.use_idloss:
        id_loss = IDLoss(args.identity_path).to(device).eval()
    else:
        id_loss = None

    train(args, dataloader, netG, netD, optimizer_G, optimizer_D, netG_ema, 
          vgg_loss, id_loss, device, dff_masks)
