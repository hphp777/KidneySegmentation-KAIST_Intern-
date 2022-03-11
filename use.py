import torch 
from unet.unet_model import UNet
from predict import predict_img, mask_to_image
from scipy import io
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL.Image as PIL
import numpy as np

def dice(result, label):
    # result, label dice coefficient
    total = 256 * 256 * 2
    intersection = 0

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            if result[y][x] == label[y][x]:
                intersection += 1
    intersection *= 2
    diceCoefficient = intersection / total
    return diceCoefficient


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resize = 256

data_dir = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/"

net = UNet(n_channels=1, n_classes=2)

net.to(device=device)
weight = "C:/Users/bispl2219/Desktop/Kidney_Segmentation/checkpoints/checkpoint_epoch4.pth"
net.load_state_dict(torch.load(weight, map_location=device))

kidney = io.loadmat(data_dir + "data/img3800.mat")
kidney = kidney['data']

kidney /= kidney.max()
kidney += 1

mean = np.mean(kidney)
std = np.std(kidney)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([resize,resize], PIL.NEAREST),
    transforms.ToTensor(),
    ])

transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([resize,resize], PIL.NEAREST),
    transforms.ToTensor(),
    ])


label = io.loadmat(data_dir + "data/mask3800.mat")
label = label['data']

input = transform(kidney)
label = transform2(label)

mask = predict_img(net=net, full_img=input, device=device)

result = mask_to_image(mask)

# Calcuate Dice Coefficient
print(dice(np.argmax(mask, axis=0), label[0]))

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 6))
ax1.imshow(input[0])
ax1.set_title('input')
ax2.imshow(result)
ax2.set_title('prediction')
ax3.imshow(label[0])
ax3.set_title('mask')
plt.savefig("C:/Users/bispl2219/Desktop/Kidney_Segmentation/checkpoints/results/37")
