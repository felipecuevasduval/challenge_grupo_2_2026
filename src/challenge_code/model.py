import torch
import torch.nn as nn
import timm

class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Conv-ReLU-MaxPooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv-ReLU-MaxPooling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # fuerza a 8x8 sin importar el tamaño de entrada
        self.adapt = nn.AdaptiveAvgPool2d((8, 8))

        # Flattening + FCs
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(512, 512)
        self.relu4 = nn.ReLU()

        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool2(self.relu2(self.conv3(x)))
        x = self.adapt(x)

        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.relu4(self.fc2(x))
        x = self.fc_out(x)
        return x
class VGG19Timm(nn.Module):

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "vgg19",
            pretrained=pretrained,
            num_classes=num_classes, 
        )

    def forward(self, x):
        return self.backbone(x)
class DinoV2Timm(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        model_name: str = "vit_small_patch14_reg4_dinov2",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = ConvolutionalNetwork(17)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # [2, 17]