"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, data_aug_transform
from model import Net, AlexNet, ResNet18,  ResNet50, DinoV2Base, DinoV2Large

class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform, self.data_augm_transform = self.init_transform()
    

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        if self.model_name == "alexnet":
            return AlexNet()
        if self.model_name == "resnet18":
            return ResNet18()
        if self.model_name == "resnet50":
            return ResNet50()
        if self.model_name == "dinov2base":
            return DinoV2Base()
        if self.model_name == "dinov2large":
            return DinoV2Large()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        return data_transforms, data_aug_transform

    
    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform
    
    def get_transform_aug(self):
        return self.data_augm_transform

    def get_all(self):
        return self.model, self.transform, self.data_augm_transform
