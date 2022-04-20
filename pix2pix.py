def forward_old(self, x):
        in_channel = 3
        out_channel = 64
        skips = []
        for i in range(8):
            if i == 0:
                x = self.encoder(in_channel, out_channel, x, False)
            else:
                x = self.encoder(in_channel, out_channel, x)
            in_channel = out_channel
            out_channel = min(512, out_channel*2)
            skips.append(x)
        skips = reverse(skips[:-1])
        for i in range(7):
            if i<3:
                x = self.decoder(in_channel, out_channel, x, False)
            else:
                x = self.decoder(in_channel, out_channel, x)
            in_channel = out_channel
            out_channel = min(512, out_channel/2) if i>3 else 512
            x = x + skips[i]
        x = self.decoder(64, 3, x)
        output = self.act(x)
        return output