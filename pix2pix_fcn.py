# %%
from glob import glob
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from os import listdir
import os
from os.path import join
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import functools
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(
    description="Training script for pix2pix",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset path
parser.add_argument("--data-dir", type=str, 
    help="Folder for datasets", 
    default="../dataset/cityscapes")
# Dataset folder (in path)
parser.add_argument("--data-seg", type=str, 
    help="Segment in datasets",
    default="night2day")
# Save paths
parser.add_argument("--out-dir", type=str, help="Output directory", 
    default="")#default="/scratch/lonelyshark99/pix2pix_zen_ada/")
# Number of epochs for training
parser.add_argument("--num-epochs", type=int, 
    help="Number of epochs for training", default=17)
# Checkpoint frequency
parser.add_argument("--epoch-ckpt-freq", type=int, 
    help="Checkpoint frequency (in epochs)", default=1)
parser.add_argument("--network", type=str, 
    help="Network", default="Unet")
parser.add_argument("--loss", type=str, 
    help="Network", default="l1cgan")
parser.add_argument("--batch", type=int, 
    help="Network", default=1)
args, uk_args = parser.parse_known_args()
print(f"Known arguments: {args}")
# %%

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        encoder_model = []
        in_channel = 3
        out_channel = 64
        encoder_block = []
        encoder = [
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
        ]
        encoder_block.append(encoder)
        encoder_model = encoder_model + encoder
        in_channel = out_channel
        out_channel = min(512, out_channel*2)
        for i in range(1,7):
            encoder = [
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_channel),
                ]
            encoder_block.append(encoder)
            encoder_model = encoder_model + encoder
            in_channel = out_channel
            out_channel = min(512, out_channel*2)
        encoder = [
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True)
        ]
        encoder_block.append(encoder)
        encoder_model = encoder_model + encoder
        in_channel = out_channel
        out_channel = min(512, out_channel*2)
        self.encoder_model = encoder_block

        decoder_model = []
        decoder_block = []
        for i in range(0, 7):
            if i<3:
                decoder = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout2d(p=0.5)
                ]
                decoder_block.append(decoder)
                decoder_model = decoder_model + decoder
            else:
                decoder = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(out_channel),
                ]
                decoder_block.append(decoder)
                decoder_model = decoder_model + decoder
            in_channel = 2*out_channel
            out_channel = min(512, int(out_channel/2)) if i>2 else 512
        decoder = [
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ]
        decoder_block.append(decoder)
        decoder_model = decoder_model + decoder
        self.decoder_model = decoder_block
        model = encoder_model + decoder_model
        self.model = nn.Sequential(*model)

    def forward(self, x):
        skips = []
        idx = 0
        for encoder in self.encoder_model:
            for mod in encoder:
                x = mod(x)
            skips.append(x)
        skips = skips[:-1]
        skips.reverse()
        decoders = self.decoder_model[:-1]
        for decoder, skip in zip(decoders, skips):
            for mod in decoder:
                x = mod(x)
            x = torch.cat((x, skip), 1)
            
        x = self.decoder_model[-1][0](x)
        x = self.decoder_model[-1][1](x)
        return x


class GeneratorED(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        encoder_model = []
        in_channel = 3
        out_channel = 64
        encoder_block = []
        encoder = [
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
        ]
        encoder_block.append(encoder)
        encoder_model = encoder_model + encoder
        in_channel = out_channel
        out_channel = min(512, out_channel*2)
        for i in range(1,7):
            encoder = [
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_channel),
                ]
            encoder_block.append(encoder)
            encoder_model = encoder_model + encoder
            in_channel = out_channel
            out_channel = min(512, out_channel*2)
        encoder = [
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True)
        ]
        encoder_block.append(encoder)
        encoder_model = encoder_model + encoder
        in_channel = out_channel
        out_channel = min(512, out_channel*2)
        self.encoder_model = encoder_block

        decoder_model = []
        decoder_block = []
        for i in range(0, 7):
            if i<3:
                decoder = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout2d(p=0.5)
                ]
                decoder_block.append(decoder)
                decoder_model = decoder_model + decoder
            else:
                decoder = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(out_channel),
                ]
                decoder_block.append(decoder)
                decoder_model = decoder_model + decoder
            in_channel = out_channel
            out_channel = min(512, int(out_channel/2)) if i>2 else 512
        decoder = [
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ]
        decoder_block.append(decoder)
        decoder_model = decoder_model + decoder
        self.decoder_model = decoder_block
        model = encoder_model + decoder_model
        self.model = nn.Sequential(*model)

    def forward(self, x):
        skips = []
        idx = 0
        for encoder in self.encoder_model:
            for mod in encoder:
                x = mod(x)
            skips.append(x)
        skips = skips[:-1]
        skips.reverse()
        decoders = self.decoder_model[:-1]
        for decoder, skip in zip(decoders, skips):
            for mod in decoder:
                x = mod(x)
            #x = torch.cat((x, skip), 1)
            
        x = self.decoder_model[-1][0](x)
        x = self.decoder_model[-1][1](x)
        return x


def Gloss_function(output, target, discriminator_op, device):
    LAMBDA = 100
    l1_loss = nn.L1Loss().to(device)
    bce_loss = nn.BCELoss().to(device)
    gan_l1_loss = l1_loss(output, target)
    gan_loss = bce_loss(discriminator_op, torch.ones_like(discriminator_op))
    loss = gan_loss + LAMBDA*gan_l1_loss

    return loss
        

# %%
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        discriminator = []
        p1 = 1
        p2 = 1
        discriminator.append(nn.Conv2d(in_channels=6, out_channels=64, kernel_size=4, stride=2, padding=1))
        discriminator.append(nn.LeakyReLU(0.2, True))
        
        discriminator.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True))
        discriminator.append(nn.BatchNorm2d(128))
        discriminator.append(nn.LeakyReLU(0.2, True))

        discriminator.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True))
        discriminator.append(nn.BatchNorm2d(256))
        discriminator.append(nn.LeakyReLU(0.2, True))

        discriminator.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=True))
        discriminator.append(nn.BatchNorm2d(512))
        discriminator.append(nn.LeakyReLU(0.2, True))

        discriminator.append(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*discriminator)
    
    def forward(self, x):
        return self.model(x)

def Dloss_function(real_image, generated_image, device):
    disc_loss = nn.BCELoss().to(device)
    loss_generated = disc_loss(generated_image, torch.zeros_like(generated_image))
    loss_real = disc_loss(real_image, torch.ones_like(real_image))
    loss = loss_generated+loss_real

    return loss
        

# %%
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        discriminator = []
        p1 = 1
        p2 = 1
        discriminator.append(nn.Conv2d(in_channels=6, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True))
        discriminator.append(nn.LeakyReLU(0.2, True))
        
        discriminator.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True))
        discriminator.append(nn.BatchNorm2d(128))
        discriminator.append(nn.LeakyReLU(0.2, True))

        discriminator.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True))
        discriminator.append(nn.BatchNorm2d(256))
        discriminator.append(nn.LeakyReLU(0.2, True))

        discriminator.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=True))
        discriminator.append(nn.BatchNorm2d(512))
        discriminator.append(nn.LeakyReLU(0.2, True))

        discriminator.append(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True))
        self.model = nn.Sequential(*discriminator)
    
    def forward(self, x):
        return self.model(x)

def Dloss_function(real_image, generated_image, device):
    disc_loss = nn.BCELoss().to(device)
    loss_generated = disc_loss(generated_image, torch.zeros_like(generated_image))
    loss_real = disc_loss(real_image, torch.ones_like(real_image))
    loss = loss_generated+loss_real

    return loss


dir_seg = "../dataset/leftImg8bit/gtFine/val/lindau"
files = glob(dir_seg+"/*")
print(len(files))
seg_file = files
print(len(seg_file), seg_file[0])

dir_sreal = "../dataset/leftImg8bit/val/munster"
files = glob(dir_sreal+"/*")
print(len(files))
real_file = files
print(len(real_file), real_file[0])

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Building Generator")
if args.network == "Unet":
    print("Unet")
    generator_model = Generator()
elif args.network == "ED":
    print("ED")
    generator_model = GeneratorED()
else:
    quit()
#generator_model.apply(init_weights)
generator_model = generator_model.to(device=device)

# %%
transform_list = []
transform_list.append(tf.ToPILImage())
transform_list += [tf.ToTensor()]
transform_list += [tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform_list = tf.Compose(transform_list)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
with torch.no_grad():
    net_g = torch.load('generator_model.pth') # m
    net_g.eval()
    for input_file in tqdm(seg_file):
        #input_img, real_img = batch[0].to('cuda'), batch[1].to('cuda')
        city, shot, frame, type, img = (os.path.basename(input_file).split('_'))
        if img != "color.png":
            continue
        #print(input_file)
        input_img = cv2.imread(input_file)#Image.open(input_img).convert('RGB')
        input_img = cv2.resize(input_img, (256, 256))
        #cv2.imshow("img", input_img)
        #cv2.waitKey(0)
        #input_img = input_img.transpose(1,2,0)
        #input_img = input_img.reshape(1, input_img.shape[0], input_img.shape[1], input_img.shape[2])
        input_img = transform_list(input_img)
        input_img = torch.reshape(input_img, (1, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
        gen_op = net_g(input_img.to('cuda'))
        gen_op = ((gen_op[0].cpu().detach().numpy().transpose(1,2,0)+1)*127.5).astype(np.uint8)
        
        name = "_".join([city, shot, frame])
        cv2.imwrite("../dataset/results/ed_l1cgan/"+name+"_leftImg8bit.png", gen_op)
        #cv2.imshow("img", gen_op)
        #cv2.waitKey(0)
        #quit()
        #gen_op = (gen_op - torch.min(gen_op))/(torch.max(gen_op)- torch.min(gen_op)) * 255
        #ax1.imshow(((input_img[0].cpu().detach().numpy().transpose(1,2,0)+1)*127.5).astype(np.uint8))
        #ax2.imshow(((real_img[0].cpu().detach().numpy().transpose(1,2,0)+1)*127.5).astype(np.uint8))
        #ax3.imshow(((gen_op[0].cpu().detach().numpy().transpose(1,2,0)+0)*1).astype(np.uint8))
        #plt.pause(1)