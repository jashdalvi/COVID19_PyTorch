import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from PIL import Image

images = np.load("/home/jash/Desktop/JashWork/Covid19CT/images_medseg.npy")
masks = np.load("/home/jash/Desktop/JashWork/Covid19CT/masks_medseg.npy")
# print(data.shape) #(100,512,512,1)
# print(masks.shape) #(100,512,512,3)

# x = np.array(np.clip(data[0],0,255),dtype = np.uint8)
def preprocess_image(image):

    return (image - image.min())/(image.max() - image.min())


class CovDataset(torch.utils.data.Dataset):
    def __init__(self,images,transform,masks):
        self.images = images
        self.transform = transform
        self.masks = masks
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self,idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = np.concatenate([image,image,image],axis = -1)
        image = np.transpose(image,(1,0,2))
        mask = np.transpose(mask,(2,1,0))
        image = preprocess_image(image)
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        image = self.transform(image)
        mask = torch.tensor(mask,dtype=torch.float32)
        mask = transforms.Resize(256)(mask)
        return image,mask

np.random.seed(42)
train_size = 0.9
idxs = np.random.permutation(100)
images = images[idxs]
masks = masks[idxs]
total_train_images = int(train_size*images.shape[0])
train_images = images[:total_train_images]
valid_images = images[total_train_images:]
train_masks = masks[:total_train_images]
valid_masks = masks[total_train_images:]


def get_train_test_loader(train_images,valid_images,train_masks,valid_masks):

    
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    
    common_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    data_augmentation = False
    if data_augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness = 0.9),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(20,fill = (0,0,0)),
            transforms.RandomCrop(224,padding = 26),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing()   
        ])
    else:
        train_transforms = common_transforms
     
    train_dataset = CovDataset(train_images,common_transforms,train_masks)
    test_dataset = CovDataset(valid_images,common_transforms,valid_masks)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 8,shuffle = True,num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 8,shuffle = False,num_workers = 4)
    
    
    return train_loader,test_loader


class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
    
    def get_model(self):
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=True)

        model.conv = torch.nn.Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1))

        return model

model = UNet().get_model()


loss_fn = nn.CrossEntropyLoss()

def get_optimizer_and_scheduler(model):
    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = 0.001,
        momentum = 0.9
    )

    decay_rate = 0.1

    lmbda = lambda epoch: 1/(1 + decay_rate * epoch)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                            factor = 0.5,
#                                                            patience = 6,
#                                                            verbose = True)
    
    return optimizer, scheduler

def train(model,train_loader,optimizer,device):
    batch_loss = np.array([])
    batch_acc = np.array([])
    
    model.train()
    for x,y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
    
        loss = loss_fn(y_pred,y)
        loss_value = loss.item()
        
        loss.backward()
        optimizer.step()
        
        # y_pred_idx = y_pred.max(dim = 1)[1]
        # accuracy = accuracy_score(y_pred_idx.cpu().numpy(),y.cpu().numpy())
        batch_loss = np.append(batch_loss,[loss_value])
        # batch_acc = np.append(batch_acc,[accuracy])
        
    return batch_loss.mean()

def valid(model,test_loader,optimizer,device):
    batch_loss = np.array([])
    batch_acc = np.array([])
    
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)

            loss = loss_fn(y_pred,y)
            # y_pred_idx = y_pred.max(dim = 1)[1]
            # accuracy = (y_pred_idx.cpu().numpy() == y.cpu().numpy()).mean()
            batch_loss = np.append(batch_loss,[loss.item()])
            # batch_acc = np.append(batch_acc,[accuracy])
        
    return batch_loss.mean()


def main(model,optimizer,scheduler):
    device = "cpu"
    train_loader, test_loader = get_train_test_loader(train_images,valid_images,train_masks,valid_masks)
    model.to(device)
    scheduler = False
    best_loss = 100000
    valid_losses = []
    train_losses = []
    for epoch in range(200):
        
        train_loss= train(model,train_loader,optimizer,device)
        valid_loss = valid(model,test_loader,optimizer,device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print("------------------------------------------------------")
        print("\nEpoch : {}\tTrain Loss: {}".format(epoch,train_loss))
        # print("Epoch : {}\tTrain Accuracy: {}".format(epoch,train_acc))
        print("Epoch : {}\tValid Loss: {}".format(epoch,valid_loss))
        # print("Epoch : {}\tValid Accuracy: {}\n".format(epoch,valid_acc))
        print("-------------------------------------------------------")
        
        if valid_loss < best_loss:
            model_path = "./unet.pt"
            torch.save(model.state_dict(),model_path)
            best_loss = valid_loss

        if scheduler:
            scheduler.step()
    return train_losses,valid_losses

optimizer, scheduler = get_optimizer_and_scheduler(model)

#Trainig the model and saving it according to the best validation accuracy
train_losses, valid_losses = main(model,optimizer,scheduler)

