import torch
import torch.nn as nn
import torchvision.models as models


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model defination
class MobileNetV2Model(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Model, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)
        self.mobilenetv2.classifier[1] = nn.Linear(self.mobilenetv2.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.mobilenetv2(x)

class InceptionV3Model(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3Model, self).__init__()
        self.inceptionv3 = models.inception_v3(pretrained=True)
        in_features = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.inceptionv3(x)

class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, modelC, weights):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.weights = torch.tensor(weights, device=device, dtype=torch.float32)

    def forward(self, x):
        outputA = self.modelA(x)
        outputB = self.modelB(x)
        outputC = self.modelC(x)

        # Check and process the output of the InceptionV3 model
        if isinstance(outputB, models.InceptionOutputs):
            outputB = outputB.logits  # Extract the main output

        output = (self.weights[0] * outputA +
                  self.weights[1] * outputB +
                  self.weights[2] * outputC)

        return output