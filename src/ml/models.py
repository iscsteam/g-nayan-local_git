import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

# model-1 for fundus or non funding images classifications
class BinaryClassifier(nn.Module):
    def __init__(self) -> None:
        super(BinaryClassifier, self).__init__()
        # Recommended: Use weights=models.ResNet18_Weights.DEFAULT for latest best weights
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class EfficientNetB3Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(EfficientNetB3Model, self).__init__()
        # Recommended: Use weights=models.EfficientNet_B3_Weights.DEFAULT for latest best weights if available
        # Or ensure 'efficientnet-b3' string corresponds to a valid pretrained model in your library version
        self.efficientnet = models.efficientnet_b3(weights=None) # Assuming custom weights are loaded later
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)


#left and right prediction model
class BinaryClassifier_left_right(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # It's generally better to load pre-trained models and modify them in the init,
        # rather than relying on from_pretrained within the class constructor if possible,
        # unless 'EfficientNet.from_pretrained' is standard for this specific library.
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        num_features = self.model._fc.in_features
        self.model._fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        # Removed logger call from here, logging should be handled by the application logic using the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
