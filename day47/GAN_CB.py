import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torchvision import transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Create data loader
path = '../input/img_align_celeba/img_align_celeba'
batch_size = 64
last_layer = 100

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# create datasets
dataset = datasets.ImageFolder(path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size)

# visualize
images, _ = next(iter(dataset))
img = make_grid(images)
plt.imshow(img)

# Size of latnet vector
nz = 100
# Filter size of generator
ngf = 64
# Filter size of discriminator
ndf = 64
# Output image channels
nc = 3


# Define class
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # second
            nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # third
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # fourth
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # last
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # second
            nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # third
            nn.Conv2d(ngf*2, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # fourth
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # last
            nn.Conv2d(ngf * 8, 1, 4, 2, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.model(x)


# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Create model
netG = Generator()
netG.apply(weights_init)
print(netG)

netD = Discriminator()
netD.apply(weights_init)
print(netD)

# move to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG.to(device)
netD.to(device)

# loss and optimizer
criterion = nn.BCELoss()

# create a fixed noise
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

optimG = optim.Adam(netG.parapeters(), lr=0.001)
optimD = optim.Adam(netD.parameters(), lr=0.001)

# Now train
total_epoch = 20
setp = 200

for epoch in range(total_epoch):

    for i, data in enumerate(dataloader):
        ############################
        ## Train Discriminator #####
        ############################
        netD.train()
        # train with real image
        optimD.zero_grad()
        images, _ = data
        # take batch size
        batch = images[0]
        # create variable
        images = Variable(images)
        # move to gpu
        images = images.to(device)
        # create fake and real labels
        real_labels = Variable(torch.ones(batch, 1)).to(device)
        fake_labels = Variable(torch.zeros(batch), 1).to(device)

        # pass real image
        real_output = netD(images)
        real_loss = criterion(real_output, real_labels)
        dx = real_output.data.mean()

        # train with fake data
        # create data
        fake = torch.randn(batch, nz).to(device)
        # generate fake image
        fake_img = netG(fake)
        # pass fake image to the model
        fake_output = netD(fake_img)
        fake_loss = criterion(fake_output, fake_labels)
        dz1 = fake_loss.data.mean()
        # calculate total loss
        loss_D = real_loss + fake_loss
        loss_D.backward()
        # update optimizer
        optimD.step()

        ############################
        ## Train Generator #####
        ############################
        netG.train()
        optimG.zero_grad()
        # generate fake
        fake = torch.randn(batch, nz).to(device)
        # generate fake image
        fake_g = netG(fake)
        output_g = netD(fake_g)
        loss_G = criterion(output_g, real_labels)
        loss_G.backward()
        optimG.step()
        dz2 = loss_G.data.mean()

        if epoch % setp == 0:
            print("Epoch: {}\t DX: {} DZ1{} DZ2{}".format(epoch, dx, dz1, dz2),
                  "\nD Loss: {} \tG loss: {}".format(loss_D.item(), loss_G.item()))

    print("Epoch: {}".format(epoch),
          "\nD Loss: {} \tG loss: {}".format(loss_D.item(), loss_G.item()))