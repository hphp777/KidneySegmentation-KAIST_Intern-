""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # self.up1 = up_conv(1024, 512)
        # self.up2 = up_conv(512, 256)
        # self.up3 = up_conv(256, 128)
        # self.up4 = up_conv(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, deep_supervision=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision

        param = [64,128,256,512,1024]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.init = DoubleConv(n_channels, param[0])
        self.down1 = Down(param[0], param[1])
        self.last1 = DoubleConv(param[0] + param[1], param[0])
        self.lasr1_1 = DoubleConv(param[1],param[0])

        self.down2 = Down(param[1], param[2])
        self.last2_1 = DoubleConv(param[1] + param[2],param[1])
        self.last2_2 = DoubleConv(param[0]*2+param[1], param[0])

        self.down3 = Down(param[2], param[3])

        self.last3_1 = DoubleConv(param[2] + param[3],param[2])
        self.last3_2 = DoubleConv(param[1]+param[1]+param[2],param[1])
        self.last3_3 = DoubleConv(param[0]*3+param[1],param[0])

        factor = 2 if bilinear else 1
        factor = 1

        self.down4 = Down(param[3], param[4] // factor)
        self.last4_1 = DoubleConv(param[3]+param[4],param[3])
        self.last4_2 = DoubleConv(param[2] + param[2] + param[3], param[2])
        self.last4_3 = DoubleConv(param[1]+param[1]+param[1]+param[2],param[1])
        self.last4_4 = DoubleConv(param[0]*4+param[1],param[0])

        if self.deep_supervision:
            self.output1 = OutConv(64, n_classes)
            self.output2 = OutConv(64, n_classes)
            self.output3 = OutConv(64, n_classes)
            self.output4 = OutConv(64, n_classes)

        else:
            self.output = OutConv(64, n_classes)


        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x0_0 = self.init(x) #(1_0) 1*256*256 -> 64*256*256 # DoubleConv

        x1_1 = self.down1(x0_0) # (2_0) 64*256*256 -> 128*128*128 # Polling -> DoubleConv
        x1_2 = self.up(x1_1) # (3_0) 128*256*256 # Upsampling
        x1_3 = torch.cat([x0_0, x1_2], dim=1) # (4_0) 192*256*256 # Concaenate (1),(3)
        x0_1 = self.last1(x1_3) # (5_0) 64*256*256 # DoubleConv

        x2_1 = self.down2(x1_1) # (1_1) 128*128*128 -> 256*64*64 # Polling -> DoubleConv
        x2_2 = self.up(x2_1) # (2_1) 256*128*128 # Upsampling
        x2_3 = torch.cat([x1_1, x2_2], dim=1) # (3_1) (256+128)*128*128 # Concatenate (x1_1),(x2_2)
        x2_4 = self.last2_1(x2_3) # (4_1) 128*128*128 # DoubleConv
        x2_5 = self.up(x2_4) # (5_1) 128*256*256 # Upsampling
        x0_2 = torch.cat([x0_0, x0_1, x2_5], dim=1) # (6_1) (64+64+128)*256*256 # concatenate (0_0),(5_0),(5_1) = 256*256*256
        x0_2 = self.last2_2(x0_2)

        x3_1 = self.down3(x2_1) # (1_2) 256*64*64 -> 512*32*32 # Polling -> DoubleConv
        x3_2 = self.up(x3_1) # (2_2) 512*32*32 -> 512*64*64 # Upsampling
        x3_3 = torch.cat([x2_1, x3_2], dim=1) # (3_2) (256+512)*64*64 # Concatenate (2_1),(3_2)
        x3_4 = self.last3_1(x3_3) # (4_2) 256*64*64 #DoubleConv
        x3_5 = self.up(x3_4) # (5_2) 256*128*128 # Upsampling
        x3_6 = torch.cat([x1_1,x2_4,x3_5], dim=1) # (5_3) (128+128+256)*128*128 # Concatenate  (x1_1), (x2_3)
        x3_7 = self.last3_2(x3_6) # 128*128*128
        x3_8 = self.up(x3_7) #128*256*256
        x0_3 = torch.cat([x0_0,x0_1,x0_2,x3_8], dim=1) #(64+64+64+128)*256*256 = 320*256*256
        x0_3 = self.last3_3(x0_3)

        x4_1 = self.down4(x3_1) # 512*32*32 -> 1024*16*16
        x4_2 = self.up(x4_1) # 1024*32*32
        x4_3 = torch.cat([x3_1, x4_2], dim=1) # (512+1024)*32*32
        x4_4 = self.last4_1(x4_3) # 512*32*32
        x4_5 = self.up(x4_4) # 512*64*64
        x4_6 = torch.cat([x2_1, x3_4, x4_5],dim = 1) # (256+256+512)*64*64
        x4_7 = self.last4_2(x4_6) # 256*64*64
        x4_8 = self.up(x4_7) # 256*128*128
        x4_9 = torch.cat([x1_1,x2_4,x3_7,x4_8], dim=1) # (128 + 128 + 128 + 256)*128*128
        x4_10 = self.last4_3(x4_9) # 128*128*128
        x4_11 = self.up(x4_10) # 128*256*256
        x0_4 = torch.cat([x0_0,x0_1,x0_2,x0_3,x4_11],dim=1) #(64+64+64+64+128)*256*256
        x0_4 = self.last4_4(x0_4) 

        if self.deep_supervision == True:
            output1 = self.output1(x0_1)
            output2 = self.output2(x0_2)
            output3 = self.output3(x0_3)
            output4 = self.output4(x0_4)
            logits = (output1 + output2 + output3 + output4) / 4
        else:
            logits = self.output(x0_4)

        return logits

class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=2):
        super(AttU_Net,self).__init__()

        self.n_channels = img_ch
        self.n_classes = output_ch
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = DoubleConv(img_ch,64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)
        self.Conv5 = DoubleConv(512, 1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = DoubleConv(1024, 512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = DoubleConv(512, 256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = DoubleConv(256, 128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = DoubleConv(128, 64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=2,t=2):
        super(R2AttU_Net,self).__init__()

        self.n_channels = img_ch
        self.n_classes = output_ch
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1