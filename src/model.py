import torch.nn as nn
import torch.nn.functional as F
import torch


class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # Load the largest DINOv2 model (ViT-g/14) from torch.hub
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        
        # Freeze backbone parameters to use as feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Add classification head for 4 classes
        self.classifier = nn.Linear(self.backbone.embed_dim, 4)

    ###########DO NOT CHANGE THIS PART##################
    def init(self):
        self.load_state_dict(torch.load("model.pth",weights_only=True))
    ####################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Input:
        - x: A 4D input tensor representing a batch of images, with shape (batch_size, channels, height, width).
            For instance, for a batch of RGB images of size 128x128, the shape would be (batch_size, 3, 128, 128).

        Output:
        - A tensor of shape (batch_size, num_classes), where `num_classes` corresponds to the number of target classes
        for classification. In this case it is 4.
        """
                # DINOv2 expects 224x224 images, so we need to resize from 128x128
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Get features from DINOv2 backbone (CLS token)
        features = self.backbone(x)  # Shape: (batch_size, embed_dim)
        
        # Apply classification head
        logits = self.classifier(features)  # Shape: (batch_size, 4)
        
        return logits


from torchvision.models import efficientnet_b3

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Use EfficientNet-B2 as the backbone (good balance of performance and size)
        self.backbone = efficientnet_b3()
        
        # Get the number of input features for the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)
        )

    def init(self):
        self.load_state_dict(torch.load("model.pth", weights_only=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EfficientNet expects 224x224 images by default
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return self.backbone(x)