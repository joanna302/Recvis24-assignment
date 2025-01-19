import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision 

from torchvision import models 

nclasses = 500

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AlexNet(nn.Module): 
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.model.classifier[-1].out_features = nclasses

    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module): 
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fc.out_features, self.model.fc.out_features), 
            nn.ReLU(), 
            nn.Linear(self.model.fc.out_features, nclasses), 
        )
   
    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)
    

class ResNet50(nn.Module): 
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fc.out_features, self.model.fc.out_features), 
            nn.ReLU(), 
            nn.Linear(self.model.fc.out_features, nclasses), 
        )
   
    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)


class DinoV2Base(nn.Module): 
    def __init__(self):
        super(DinoV2Base, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.norm.normalized_shape[0], 512), 
            nn.ReLU(), 
            nn.Linear(512, nclasses), 
        )
        
    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)

class DinoV2Large(nn.Module): 
    def __init__(self):
        super(DinoV2Large, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.norm.normalized_shape[0], self.model.norm.normalized_shape[0]), 
            nn.ReLU(), 
            nn.Linear(self.model.norm.normalized_shape[0], 512), 
            nn.ReLU(), 
            nn.Linear(512, nclasses), 
        )
        

    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)
