# %%
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from os import listdir
from os.path import join
from PIL import Image
#from tqdm import tqdm
#from matplotlib import pyplot as plt
import functools
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Training script for pix2pix",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset path
parser.add_argument("--data-dir", type=str, 
    help="Folder for datasets", 
    default="~/Downloads/Datasets/pix2pix")
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
args, uk_args = parser.parse_known_args()
print(f"Known arguments: {args}")
# %%
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.path = image_dir
        self.image_filenames = [x for x in listdir(self.path)]
        transform_list = []
        transform_list.append(tf.ToPILImage())
        transform_list.append(tf.Resize((286, 286), interpolation=tf.InterpolationMode.BICUBIC))
        transform_list.append(tf.RandomCrop((256, 256)))
        transform_list.append(tf.RandomHorizontalFlip())
        transform_list += [tf.ToTensor()]
        transform_list += [tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform_list = tf.Compose(transform_list)
        self.tensor_tf = tf.ToTensor()

    def __getitem__(self, index):
        a = Image.open(join(self.path, self.image_filenames[index])).convert('RGB')
        a = np.asarray(a)
        w = a.shape[1]
        w = w // 2
        input_image = a[:, w:, :]
        real_image = a[:, :w, :]
        #input_image = self.tensor_tf(input_image)
        #real_image = self.tensor_tf(real_image)
        #input_image = torch.from_numpy(input_image)
        #real_image = torch.from_numpy(real_image)
        A = self.transform_list(input_image)
        B = self.transform_list(real_image)
        return A, B

        input_image = self.tensor_tf(input_image)
        real_image = self.tensor_tf(real_image)
        #input_image, real_image = random_jitter(input_image, real_image)
        input_image, real_image = normalize(input_image, real_image)
        """#plt.imshow(input_image.cpu().detach().numpy().transpose(1,2,0))
        #plt.show()
        #plt.pause(5)
        #plt.imshow(real_image.cpu().detach().numpy().transpose(1,2,0))
        #plt.show()
        #plt.pause(10)
        quit()"""

        return input_image, real_image
        

    def __len__(self):
        return len(self.image_filenames)
# %%
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, pred, target):
        if target:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(pred)

    def __call__(self, pred, target):
        target_tensor = self.get_target_tensor(pred, target)
        loss = self.loss(pred, target_tensor)
        return loss

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        encoder_model = []
        in_channel = 3
        out_channel = 64
        encoder_block = []
        for i in range(8):
            if i > 0 and i<7:
                encoder = [
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(out_channel),
                    nn.LeakyReLU(0.2, True)
                ]
                encoder_block.append(encoder)
                encoder_model = encoder_model + encoder
            elif i==7:
                encoder = [
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.LeakyReLU(0.2, True)
                ]
                encoder_block.append(encoder)
                encoder_model = encoder_model + encoder
            else:
                encoder = [
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.LeakyReLU(0.2, True)
                ]
                encoder_block.append(encoder)
                encoder_model = encoder_model + encoder
            in_channel = out_channel
            out_channel = min(512, out_channel*2)
        self.encoder_model = encoder_block
        decoder_model = []
        decoder_block = []
        decoder = [
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(p=0.5),
            nn.ReLU(True)
        ]
        decoder_block.append(decoder)
        decoder_model = decoder_model + decoder
        in_channel*=2
        for i in range(1, 7):
            if i<3:
                decoder = [
                    nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(out_channel),
                    nn.Dropout2d(p=0.5),
                    nn.ReLU(True)
                ]
                decoder_block.append(decoder)
                decoder_model = decoder_model + decoder
            else:
                decoder = [
                    nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True)
                ]
                decoder_block.append(decoder)
                decoder_model = decoder_model + decoder
            in_channel = 2*out_channel
            out_channel = min(512, int(out_channel/2)) if i>2 else 512
        decoder = [
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True),
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

"""
class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        
        #Paper details:
        #- C64-C128-C256-C512-C512-C512-C512-C512
        #- All convolutions are 4×4 spatial filters applied with stride 2
        #- Convolutions in the encoder downsample by a factor of 2
        
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x
    
class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x

class Generator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        
        #Paper details:
        #- Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        #- All convolutions are 4×4 spatial filters applied with stride 2
        #- Convolutions in the encoder downsample by a factor of 2
        #- Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return self.tanh(x)


class Discriminator(nn.Module):

    def __init__(self, input_channels=6):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn
"""
# %%
def resize(input_image, real_image, height, width):
    res = tf.Resize((height, width), interpolation=tf.InterpolationMode.NEAREST)
    input_image = res(input_image)
    real_image = res(real_image)
    return input_image, real_image

def random_crop(input_image, real_image):
    rc = tf.RandomCrop((256, 256))
    stacked_image = torch.cat([input_image, real_image], axis=0)
    cropped_image = rc(stacked_image)
    return cropped_image[0:3], cropped_image[3:]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if torch.rand(1,1) > 0.5:
        # Random mirroring
        hf = tf.RandomHorizontalFlip()
        input_image = hf(input_image)
        real_image = hf(real_image)
    
    """#plt.imshow(input_image.cpu().detach().numpy().transpose(1,2,0))
    #plt.show()
    #plt.pause(5)
    #plt.imshow(real_image.cpu().detach().numpy().transpose(1,2,0))
    #plt.show()
    #plt.pause(10)"""

    return input_image, real_image

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


# %%
train_dir = args.data_dir+"/train"  #../dataset/night2day/train/"
train_data = DatasetFromFolder(train_dir)

#val_dir = "../dataset/night2day/val/"
#val_data = DatasetFromFolder(val_dir)

#test_dir = "../dataset/night2day/test/"
#test_data = DatasetFromFolder(test_dir)

# %%

training_data_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
#validation_data_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)
#testing_data_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False)

# %%

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

print("Building Generator")
generator_model = Generator()
generator_model.apply(init_weights)
generator_model = generator_model.to(device=device)
print(generator_model)

# %%

discriminator_model = Discriminator()
discriminator_model.apply(init_weights)
discriminator_model = discriminator_model.to(device=device)
print(discriminator_model)

# %%

gan_optim = torch.optim.Adam(generator_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optim = torch.optim.Adam(discriminator_model.parameters(), lr=0.0002, betas=(0.5, 0.999))



# %%

epochs = int(args.num_epochs)
#fig, (ax1, ax2, ax3) = #plt.subplots(1, 3)
criterionGAN = GANLoss().to(device)
criterionL1 = torch.nn.L1Loss()
bce_loss = nn.BCEWithLogitsLoss() 
l1_loss = nn.L1Loss()
train = True
torch.autograd.set_detect_anomaly(True)
if train:
    for i in range(epochs):
        #plt.clf()
        #plt.ion()
        gl = []
        dl = []
        for iteration, batch in enumerate(training_data_loader, 1):
            input_img, real_img = batch[0].to(device), batch[1].to(device)

            fake_img = generator_model(input_img)

            # Generator updates
            #for param in discriminator_model.parameters():
            #    param.requires_grad = False
            gan_optim.zero_grad()
            fake_input = torch.cat((input_img, fake_img), 1)
            pred_fake = discriminator_model.forward(fake_input)
            """gan_l1_loss = criterionL1(fake_img, real_img) * 100
            gan_loss = criterionGAN(pred_fake, True)"""
            gan_l1_loss = l1_loss(fake_img, real_img) * 100.0
            gan_loss = bce_loss(pred_fake, torch.ones_like(pred_fake))
            loss_g = gan_loss + gan_l1_loss
            loss_g.backward()
            gan_optim.step()

            # Discriminator updates
            #for param in discriminator_model.parameters():
            #    param.requires_grad = True
            disc_optim.zero_grad()
            """real_input = torch.cat((input_img, real_img), 1)
            pred_real = discriminator_model(real_input)
            loss_real = criterionGAN(pred_real, True)

            fake_input = torch.cat((input_img, fake_img), 1)
            pred_fake = discriminator_model(fake_input.detach())
            loss_generated = criterionGAN(pred_fake, False)"""

            real_input = torch.cat((input_img, real_img), 1)
            pred_real = discriminator_model.forward(real_input)
            loss_real = bce_loss(pred_real, torch.ones_like(pred_real))

            fake_input = torch.cat((input_img, fake_img.detach()), 1)
            pred_fake = discriminator_model.forward(fake_input)
            loss_generated = bce_loss(pred_fake, torch.zeros_like(pred_fake))
            
            loss_d = loss_generated+loss_real

            loss_d.backward()
            disc_optim.step()
            #gl.append(loss_g.item())
            #dl.append(loss_d.item())
        if iteration%20 == 0:
            print("===> Epoch[{}]({}/{}): Disc_Loss: {:.4f} Gen_Loss: {:.4f}".format(
                i, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

        if i%int(args.epoch_ckpt_freq) == 0:
            net_g_model_out_path = args.out_dir+"generator_model_{}.pth".format(i)
            net_d_model_out_path = args.out_dir+"discriminator_model_{}.pth".format(i)
            torch.save(generator_model, net_g_model_out_path)
            torch.save(discriminator_model, net_d_model_out_path)
        #quit()
    
    
# %%
"""with torch.no_grad():
    net_g = torch.load('generator_model.pth') # m
    net_g.eval()
    for iteration, batch in tqdm(enumerate(validation_data_loader, 1)):
        input_img, real_img = batch[0].to('cuda'), batch[1].to('cuda')
        gen_op = net_g(input_img)
        gen_op = (gen_op - torch.min(gen_op))/(torch.max(gen_op)- torch.min(gen_op)) * 255
        ax1.imshow(((input_img[0].cpu().detach().numpy().transpose(1,2,0)+1)*127.5).astype(np.uint8))
        ax2.imshow(((real_img[0].cpu().detach().numpy().transpose(1,2,0)+1)*127.5).astype(np.uint8))
        ax3.imshow(((gen_op[0].cpu().detach().numpy().transpose(1,2,0)+0)*1).astype(np.uint8))
        #plt.pause(1)"""