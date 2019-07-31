import torch


class MY_Dataset(torch.utils.data.Dataset):

    def __init__(self, feat, labels):
        super(MY_Dataset, self).__init__()
        self.conv_feat = feat
        self.labels = labels

    def __len__(self):
        return len(self.conv_feat)

    def __getitem__(self, item):
        return self.conv_feat[item], self.labels[item]


from torchvision import datasets, transforms, models

vgg = models.vgg16(pretrained=True)

class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


conv_out = LayerActivation(vgg.features, 0)
o = vgg(img.cuda())
conv_out.remove()

act =conv_out.features