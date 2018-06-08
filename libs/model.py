import torch
import torch.nn as nn
import torchvision.models as models

resnet34 = models.resnet34(pretrained=True)

# resnet 34 has 0-9, which is 10 immediate children modules
# and let's only freeze 0-4

def freezeResnet():
    for num, child in enumerate(resnet34.children()):
        # freeze the lower layers only
        #print(num)
        if num < 4:
            for param in child.parameters():
                param.requires_grad = False

freezeResnet()

transConv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
        kernel_size=3, stride=2, padding=1, output_padding=1)
transConv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
        kernel_size=3, stride=2, padding=1, output_padding=1)
transConv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
        kernel_size=3, stride=2, padding=1, output_padding=1)
transConv4 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
        kernel_size=3, stride=2, padding=1, output_padding=1)
transConv5 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
        kernel_size=7, stride=2, padding=3, output_padding=1)

# joint prediction Conv Layer, filter size 3 x 3, 21 filters
# VNect's 2D heatmap is generated from res4d,
# and location maps from res5a, we try something simpler
jointPrediction = nn.Conv2d(in_channels=128, out_channels=21,
        kernel_size=3, stride=1, padding=1)
# layers to generate the location maps
conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,
        stride=1, padding=1)
conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,
        stride=1, padding=1)

# We should predict a location map for each of x, y, z once per image
conv8 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1,
        stride=1, padding=0)
locationPrediction = nn.Conv2d(in_channels=8, out_channels=3,
        kernel_size=15, stride=1, padding=7)

# stick a bunch of ReLu non-linearity in the upsampling pipeline
model = nn.Sequential(
    *list(resnet34.children())[:-3],
    transConv2,
    nn.ReLU(),
    nn.Dropout(),
    transConv3,
    nn.ReLU(),
    nn.Dropout(),
    transConv4,
    nn.ReLU(),
    nn.Dropout(),
    transConv5,
    nn.ReLU(),
    nn.Dropout(),
)

modelHeatmap = nn.Sequential(
    jointPrediction
)

modelLocmap = nn.Sequential(
    conv6,
    nn.ReLU(),
    nn.Dropout(),
    conv7,
    nn.ReLU(),
    conv8,
    nn.ReLU(),
    locationPrediction
)


norm5d = nn.InstanceNorm1d(num_features=5)
#norm5d = nn.BatchNorm1d(num_features=5)

fc = nn.Sequential(
    nn.Linear(in_features=105, out_features=50),
    nn.ReLU(),
    nn.Linear(in_features=50, out_features=50),
    nn.ReLU(),
    #nn.Dropout(),
    nn.Linear(in_features=50, out_features=10),
    nn.Softmax(dim=1)
)









