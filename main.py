import torch 
from unet.unet_model import *
from train import train_net

batch_size = 4
learning_rate = 0.001
resize = 256
epochs = 5
weight = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/checkpoints/checkpoint_epoch10.pth"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# parameter: (input_channel, output_channel) -> (gray_scale, kidneyOrNot)
net = UNet(1, 2, bilinear=False) 
# net = UNetPlusPlus(1,2)
# net = AttU_Net() 
# net = R2AttU_Net()

# net.load_state_dict(torch.load(weight, map_location=device))

net = net.to(device)

train_net(net, device, epochs, batch_size)