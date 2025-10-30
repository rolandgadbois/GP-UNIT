import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_network import BaseNetwork

# The code is developed based on SPADE 
# https://github.com/NVlabs/SPADE/blob/master/models/networks/

class ConceptSkipLayer(nn.Module):
    def __init__(self, dff_mask: torch.Tensor):
        """
        dff_mask: Tensor of shape [C], where C = number of channels in the skip layer.
                  Should contain values in [0,1] for each channel.
        """
        super().__init__()
        self.register_buffer("dff_mask", dff_mask.view(1, -1, 1, 1))  # broadcastable
        self.alpha = 1.0  # weight for skip influence; you can tune or make it configurable

    def forward(self, f_dec: torch.Tensor, f_enc: torch.Tensor):
        """
        f_dec: decoder feature map
        f_enc: encoder feature map (skip connection)
        """
        # Ensure shape compatibility (e.g., by interpolation if needed)
        if f_dec.shape[-2:] != f_enc.shape[-2:]:
            f_enc = nn.functional.interpolate(f_enc, size=f_dec.shape[-2:], mode='bilinear', align_corners=False)
        
        # Fuse according to DFF mask
        f_out = (1 - self.alpha * self.dff_mask) * f_dec + (self.alpha * self.dff_mask) * f_enc
        return f_out
    
class ResnetBlock(BaseNetwork):
    def __init__(self, fin, fout):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = fout

        self.conv_0 = nn.Conv2d(fin, fmiddle, 3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, 3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)

        self.norm_0 = nn.InstanceNorm2d(fin, affine=True)
        self.norm_1 = nn.InstanceNorm2d(fmiddle, affine=True)
        if self.learned_shortcut:
            self.norm_s = nn.InstanceNorm2d(fin, affine=True)

    def forward(self, x):
        x_s = self.shortcut(x)

        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.conv_1(self.actvn(self.norm_1(dx)))

        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    
class DynamicSkipLayer(BaseNetwork):
    def __init__(self, hidden_nc, feature_nc):
        super(DynamicSkipLayer, self).__init__()  
        self.up = nn.Upsample(scale_factor=2)
        # Wh
        self.trans = nn.Conv2d(hidden_nc, feature_nc, 3, padding=1, padding_mode='reflect')
        # Wr
        self.reset = nn.Conv2d(feature_nc*2, feature_nc, 3, padding=1, padding_mode='reflect')
        # Wm
        self.mask = nn.Conv2d(feature_nc*2, feature_nc, 3, padding=1, padding_mode='reflect')
        # WE
        self.update = nn.Conv2d(feature_nc*2, feature_nc, 3, padding=1, padding_mode='reflect')
        # sigma
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, fdec, fenc, s):
        # h^ = sigma(Wh * up(h))
        state = F.leaky_relu(self.trans(self.up(s)), 2e-1)
        # r = sigma(Wr * [h^, fE])
        reset_gate = self.sigmoid(self.reset(torch.cat((state,fenc),dim=1)))
        # h = rh^
        new_state = reset_gate * state
        # m = sigma(Wm * [h^, fE]) with sigma=None
        mask = self.mask(torch.cat((state,fenc),dim=1))
        # apply relu + tanh to set most of the elements to zeros
        mask = (F.relu(mask)).tanh()
        # fE^ = sigma(WE * [h, fE]) with sigma=None
        new_fenc = self.update(torch.cat((new_state,fenc),dim=1))
        # f = (1-m) * fG + m * fE^
        output = (1-mask) * fdec + mask * new_fenc
        return output, new_state, mask


class GeneratorLayer(BaseNetwork):
    def __init__(self, fin, fout, content_nc, usepost=False, useskip=False):
        super().__init__()
        self.reslayer = ResnetBlock(fin, fout)
        self.postlayer = ResnetBlock(fout, fout) if usepost else None
        self.rgblayer = nn.Conv2d(fout, 3, 7, padding=3)
        if useskip:
            if dff_mask is None:
                raise ValueError("useskip=True but no dff_mask provided.")
            self.skiplayer = ConceptSkipLayer(dff_mask)  # <- replaces DynamicSkipLayer
        else:
            self.skiplayer = None

    def forward(self, x, content=None, state=None):
        mask = None
        new_state = None
        x = self.reslayer(x)
        if self.postlayer is not None:
            x = self.postlayer(x)
        rgb = self.rgblayer(F.leaky_relu(x, 2e-1))
        if self.skiplayer is not None and content is not None and state is not None:
            x = self.skiplayer(x, state)  # <- only returns fused feature
            new_state = state
        return x, rgb, new_state, mask

class Generator(BaseNetwork):
    def __init__(self, content_nc=[1,1,512,256,128,64], ngf=64):
        super().__init__()

        sequence = []
        sequence.append(GeneratorLayer(1, 8*ngf, content_nc[0], usepost=True))
        sequence.append(GeneratorLayer(8*ngf, 8*ngf, content_nc[1], useskip=True, dff_mask=dff_masks[3]))
        sequence.append(GeneratorLayer(8*ngf, 4*ngf, content_nc[2], useskip=True, dff_mask=dff_masks[2]))
        sequence.append(GeneratorLayer(4*ngf, 2*ngf, content_nc[3], useskip=True, dff_mask=dff_masks[1]))
        sequence.append(GeneratorLayer(2*ngf, 1*ngf, content_nc[4], useskip=True, dff_mask=dff_masks[0]))
        sequence.append(GeneratorLayer(1*ngf, 1*ngf, content_nc[5]))

        self.model = nn.Sequential(*sequence)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, content, scale=5, useskip=True):
        masks = []
        state = content[0] if useskip else None

        x, rgb, _, _ = self.model[0](content[0])
        if scale == 0:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, state, mask = self.model[1](x, content[2], state)
        if mask is not None:
            masks.append(mask)
        if scale == 1:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, state, mask = self.model[2](x, content[3], state)
        if mask is not None:
            masks.append(mask)
        if scale == 2:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, state, mask = self.model[3](x, content[4], state)
        if mask is not None:
            masks.append(mask)
        if scale == 3:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, state, mask = self.model[4](x, content[5], state)
        if mask is not None:
            masks.append(mask)
        if scale == 4:
            return torch.tanh(rgb), masks

        x = self.up(x)
        x, rgb, _, _ = self.model[5](x)

        return torch.tanh(rgb), masks
