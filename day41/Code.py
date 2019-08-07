import os
from PIL import Image
import torch

train_dir = ""

# check train first
trains = os.listdir(train_dir)
for i in trains:
    j = os.listdir(train_dir + "/" + i)
    for file in j:
        path = train_dir + "/" + i + file
        try:
            array = Image.open(path)
        except:
            print(path)
        break

import random

random.randint(0, 1)

from torchvision import models, transforms


train = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(256),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def save_check_point(model, epoch, classes, optimizer, scheduler=None,
                     path=None, name='model.pt'):
    try:
        classifier = model.classifier
    except AttributeError:
        classifier = model.fc

    checkpoint = {
        'class_to_name': classes,
        'epochs': epoch,
        'classifier': classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if path is None:
        d = model
    else:
        d = path + "/" + name

    torch.save(checkpoint, d)
    print(f"Model saved at {d}")


save_check_point(model, 10, classes, optimizer, scheduler=scheduler,
                     path=None, name='saved_model.pt')
