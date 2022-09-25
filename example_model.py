from base.base_model import BaseModel
from torchvision.models import resnet50, ResNet50_Weights


class AwesomeModel(BaseModel):
    def __init__(self):
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    def forward(self,x):
        return self.net(x)
