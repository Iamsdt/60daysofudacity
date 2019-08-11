from torchvision import models
model = models.densenet121(pretrained=True)

for pram in model.parameters():
    pram