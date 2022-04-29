dataset_dir = "/home/jash/Desktop/JashWork/Covid19CT/Dataset"


from imutils import paths
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
print(torchvision.__version__)
from torchvision import models
import torch.optim as optim
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class CovDataset(torch.utils.data.Dataset):
    def __init__(self,imagepaths,transform,label_mapping):
        self.imagepaths = imagepaths
        self.transform = transform
        self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self,idx):
        image = Image.open(self.imagepaths[idx]).convert("RGB")
        label = self.label_mapping[self.imagepaths[idx].split(os.path.sep)[-2]]
        image = self.transform(image)

        return image,label

class CovModel(nn.Module):
    def __init__(self):
        super(CovModel,self).__init__()

    def get_model(self):
        resnet_model = models.efficientnet_b0(pretrained=True)
        for name,param in resnet_model.named_parameters():
            param.requires_grad = True
        
        resnet_model.classifier = nn.Sequential(nn.Linear(1280,256),
                                        nn.ReLU(inplace = True),
                                        nn.Dropout(0.5),
                                        nn.Linear(256,2))
        
        return resnet_model

imagepaths = list(paths.list_images(dataset_dir))
label_mapping = {"NonCOVID" : 0, "COVID" : 1}
#Using basic train test split

np.random.seed(42)
train_size = 0.8
idxs = np.random.permutation(len(imagepaths))
imagepaths = np.array(imagepaths)[idxs].tolist()
total_train_images = int(0.8*len(imagepaths))
train_imagepaths = imagepaths[:total_train_images]
valid_imagepaths = imagepaths[total_train_images:]


#getting the train and test data loaders
def get_train_test_loader(train_imagepaths):

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    common_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    data_augmentation = True
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
     
    train_dataset = CovDataset(train_imagepaths,train_transforms,label_mapping)
    test_dataset = CovDataset(valid_imagepaths,common_transforms,label_mapping)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 32,shuffle = True,num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 32,shuffle = False,num_workers = 4)
    
    
    return train_loader,test_loader





model = CovModel().get_model()


loss = nn.CrossEntropyLoss()

def get_optimizer_and_scheduler(model):
    # optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr = 0.001,
        momentum = 0.9
    )

    decay_rate = 0.1

    lmbda = lambda epoch: 1/(1 + decay_rate * epoch)

    # Scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
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
        
        y_pred_idx = y_pred.max(dim = 1)[1]
        accuracy = accuracy_score(y_pred_idx.cpu().numpy(),y.cpu().numpy())
        batch_loss = np.append(batch_loss,[loss_value])
        batch_acc = np.append(batch_acc,[accuracy])
        
    return batch_loss.mean(),batch_acc.mean()

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
            y_pred_idx = y_pred.max(dim = 1)[1]
            accuracy = (y_pred_idx.cpu().numpy() == y.cpu().numpy()).mean()
            batch_loss = np.append(batch_loss,[loss.item()])
            batch_acc = np.append(batch_acc,[accuracy])
        
    return batch_loss.mean(),batch_acc.mean()


def main(model,optimizer,scheduler):
    
    train_loader, test_loader = get_train_test_loader(train_imagepaths,valid_imagepaths)
    model.to(device)
    
    best_acc = 0.0
    for epoch in range(50):
        
        train_loss,train_acc = train(model,train_loader,optimizer,device)
        valid_loss,valid_acc = valid(model,test_loader,optimizer,device)
        
        print("------------------------------------------------------")
        print("\nEpoch : {}\tTrain Loss: {}".format(epoch,train_loss))
        print("Epoch : {}\tTrain Accuracy: {}".format(epoch,train_acc))
        print("Epoch : {}\tValid Loss: {}".format(epoch,valid_loss))
        print("Epoch : {}\tValid Accuracy: {}\n".format(epoch,valid_acc))
        print("-------------------------------------------------------")
        
        if valid_acc > best_acc:
            model_path = "./efficientnet.pt"
            torch.save(model.state_dict(),model_path)
            best_acc = valid_acc

        if scheduler:
            scheduler.step()


optimizer, scheduler = get_optimizer_and_scheduler(model)

#Trainig the model and saving it according to the best validation accuracy
main(model,optimizer,scheduler)


