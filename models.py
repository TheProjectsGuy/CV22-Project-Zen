# %%
import torch
import torch.nn as nn
# %%
class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ##  ENCODER NETWORK
        #size = 256x256x3
        self.encoder_c64 = nn.Conv2d(in_channel=3, out_channel=64, kernel_size=4, stride=2)
        #size = 128x128x64
        self.encoder_128 = nn.Conv2d(in_channel=64, out_channel=128, kernel_size=4, stride=2)
        #size = 64x64x128
        self.encoder_256 = nn.Conv2d(in_channel=128, out_channel=256, kernel_size=4, stride=2)
        #size = 32x32x256
        self.encoder_512_1 = nn.Conv2d(in_channel=256, out_channel=512, kernel_size=4, stride=2)
        #size = 16x16x512
        self.encoder_512_2 = nn.Conv2d(in_channel=512, out_channel=512, kernel_size=4, stride=2)
        #size = 8x8x512
        self.encoder_512_3 = nn.Conv2d(in_channel=512, out_channel=512, kernel_size=4, stride=2)
        #size = 4x4x512
        self.encoder_512_4 = nn.Conv2d(in_channel=512, out_channel=512, kernel_size=4, stride=2)
        #size = 2x2x512
        self.encoder_512_5 = nn.Conv2d(in_channel=512, out_channel=512, kernel_size=4, stride=2)
        #size = 1x1x512

        ##  DECODER NETWORK
        #size = 1x1x512
        self.decoder_512_4 = nn.ConvTranspose2d(in_channel=512, out_channel=512, kernel_size=4, stride=2)
        #size = 2x2x512
        self.decoder_512_3 = nn.ConvTranspose2d(in_channel=512, out_channel=512, kernel_size=4, stride=2)
        #size = 4x4x512
        self.decoder_512_2 = nn.ConvTranspose2d(in_channel=512, out_channel=512, kernel_size=4, stride=2)
        #size = 8x8x512
        self.decoder_512_1 = nn.ConvTranspose2d(in_channel=512, out_channel=512, kernel_size=4, stride=2)
        #size = 16x16x512
        self.decoder_256 = nn.ConvTranspose2d(in_channel=512, out_channel=256, kernel_size=4, stride=2)
        #size = 32x32x256
        self.decoder_128 = nn.ConvTranspose2d(in_channel=256, out_channel=128, kernel_size=4, stride=2)
        #size = 64x64x128
        self.decoder_64 = nn.ConvTranspose2d(in_channel=128, out_channel=64, kernel_size=4, stride=2)
        #size = 128x128x64

        ## OUTPUT IMAGE
        self.output_image = nn.ConvTranspose2d(in_channel=64, out_channel=3, kernel_size=4, stride=2)
        
