import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import cv2
import torchvision.models as models
import matplotlib.pyplot as plt

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
    
    def get_model(self):
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=True)

        model.conv = torch.nn.Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1))

        return model

model = UNet().get_model()
model.load_state_dict(torch.load("/home/jash/Desktop/JashWork/COVIDCT_Flask/models/unet.pt", map_location=torch.device("cpu")))

common_transforms = transforms.Compose([
transforms.Resize((256,256)),
transforms.ToTensor()
])

# img = cv2.imread(filepath)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = img.astype(np.float32)/255.0
# img = np.transpose(img, (2, 0, 1))
filepath = "Dataset/COVID/PIIS0140673620301549_0%1.png"
img = Image.open(filepath).convert("RGB")
x = common_transforms(img)
x = x.unsqueeze(0)
model.eval()
with torch.no_grad():
    print(x.shape)
    pred = model(x)
    print(pred[0][0,:,:].sum())
    print(pred[0][1,:,:].sum())
    print(pred[0][2,:,:].sum())
    print(pred[0][3,:,:].sum())
    _,pred = pred.max(dim =1)
    print(pred.shape,pred.dtype,pred.sum()/(256*256))
    pred = pred.numpy().astype(np.uint8)
    pred = pred.transpose((1,2,0))

# pred_color = cv2.applyColorMap(pred, cv2.COLORMAP_COOL)

# pred_color = (pred.astype(np.float32)*(255/4)).astype(np.unit8)
pred_color = np.zeros((256,256,3),dtype = np.uint8)
for i in range(pred_color.shape[0]):
    for j in range(pred_color.shape[1]):
        if pred[i,j] == 3:
            pred_color[i,j,:] = np.array([0,255,0],dtype = np.uint8)
        elif pred[i,j] == 2:
            pred_color[i,j,:] = np.array([255,0,0],dtype = np.uint8)
        elif pred[i,j] == 1:
            pred_color[i,j,:] = np.array([0,0,255],dtype = np.uint8)
        else:
            pred_color[i,j,:] = np.array([255,255,0],dtype = np.uint8)
image = (x[0].numpy().transpose((1,2,0))*255.0).astype(np.uint8)
print(image.shape)
final_image = cv2.addWeighted(image,0.5,pred_color,0.5,0.0)
plt.imshow(final_image)
plt.show()