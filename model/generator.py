import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_network import BaseNetwork

# The code is developed based on SPADE 
# https://github.com/NVlabs/SPADE/blob/master/models/networks/

class ConceptSkipLayer(nn.Module):
    """
    Concept-driven skip connection layer that fuses encoder and decoder
    features using precomputed semantic concept masks (from DFF decomposition).
    Replaces the DynamicSkipLayer used in GP-UNIT.
    """

    def __init__(self, fin_enc, fin_dec, concept_mask, alpha=0.5, use_norm=True):
        super().__init__()
        self.alpha = alpha
        self.use_norm = use_norm

        # Align encoder and decoder channel dimensions
        if fin_enc != fin_dec:
            self.align = nn.Conv2d(fin_enc, fin_dec, kernel_size=1, padding=0)
        else:
            self.align = nn.Identity()

        if use_norm:
            self.norm_enc = nn.InstanceNorm2d(fin_enc, affine=False)
            self.norm_dec = nn.InstanceNorm2d(fin_dec, affine=False)

        # Register concept mask as a non-trainable buffer
        self.register_buffer("concept_mask", concept_mask.view(1, -1, 1, 1))

    def forward(self, fdec, fenc):
        fenc = self.align(fenc)

        if self.use_norm:
            fenc = self.norm_enc(fenc)
            fdec = self.norm_dec(fdec)

        f_out = (1 - self.alpha * self.concept_mask) * fdec + \
                (self.alpha * self.concept_mask) * fenc
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
        self.skiplayer = DynamicSkipLayer(content_nc, fout) if useskip else None

    def forward(self, x, content=None, state=None):
        mask = None
        new_state = None
        x = self.reslayer(x)
        if self.postlayer is not None:
            x = self.postlayer(x)
        rgb = self.rgblayer(F.leaky_relu(x, 2e-1))
        if self.skiplayer is not None and content is not None and state is not None:
            x, new_state, mask = self.skiplayer(x, content, state)
        return x, rgb, new_state, mask

class Generator(nn.Module):
    def __init__(self, dff_masks, alpha=0.6):
        """
        dff_masks: list of 6 channel-wise DFF concept masks (torch tensors)
        alpha: interpolation strength for concept-preserving skips
        """
        super().__init__()

        # --- Encoder (example channel sizes, adjust to match your GP-UNIT) ---
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.enc5 = nn.Conv2d(512, 512, 4, 2, 1)
        self.enc6 = nn.Conv2d(512, 512, 4, 2, 1)

        # --- Decoder ---
        self.dec6 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.dec5 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.dec4 = nn.ConvTranspose2d(1024, 256, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.dec1 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

        # --- Concept-aware skip layers ---
        # Each aligns decoder and encoder channels for concept-driven fusion
        self.skips = nn.ModuleList([
            ConceptSkipLayer(512, 512, dff_masks[5], alpha=alpha),  # between enc6-dec6
            ConceptSkipLayer(512, 512, dff_masks[4], alpha=alpha),
            ConceptSkipLayer(512, 256, dff_masks[3], alpha=alpha),
            ConceptSkipLayer(256, 128, dff_masks[2], alpha=alpha),
            ConceptSkipLayer(128, 64,  dff_masks[1], alpha=alpha),
            ConceptSkipLayer(64,  3,   dff_masks[0], alpha=alpha)
        ])

    def forward(self, x):
        # --- Encoder forward ---
        e1 = F.leaky_relu(self.enc1(x), 0.2)
        e2 = F.leaky_relu(self.enc2(e1), 0.2)
        e3 = F.leaky_relu(self.enc3(e2), 0.2)
        e4 = F.leaky_relu(self.enc4(e3), 0.2)
        e5 = F.leaky_relu(self.enc5(e4), 0.2)
        e6 = F.leaky_relu(self.enc6(e5), 0.2)

        # --- Decoder with concept-preserving skips ---
        d6 = F.relu(self.dec6(e6))
        d6 = self.skips[0](d6, e6)

        d5 = F.relu(self.dec5(torch.cat([d6, e5], dim=1)))
        d5 = self.skips[1](d5, e5)

        d4 = F.relu(self.dec4(torch.cat([d5, e4], dim=1)))
        d4 = self.skips[2](d4, e4)

        d3 = F.relu(self.dec3(torch.cat([d4, e3], dim=1)))
        d3 = self.skips[3](d3, e3)

        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        d2 = self.skips[4](d2, e2)

        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], dim=1)))
        d1 = self.skips[5](d1, e1)

        return d1
