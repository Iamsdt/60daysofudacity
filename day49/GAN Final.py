import os
# %matplotlib inline
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


##############################################
##  Prepare Data Loader
##############################################
path = '../input/'
image_size = 256
batch_size = 128
num_workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset
dataset = datasets.ImageFolder(root=path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
# Create the dataloader
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)


##############################################
##   Visualize Data
##############################################
images, _ = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
grid = make_grid(images.to(device)[:64], padding=2, normalize=True).cpu()
img = np.transpose(grid, (1, 2, 0))
plt.imshow(img)


##############################################
## custom weights initialization for models
##############################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


##############################################
## Discriminator
##############################################
class Discriminator(nn.Module):
    def __init__(self, nf, input_channel=3):
        super(Discriminator, self).__init__()
        self.nf = nf
        self.input_channel = input_channel

        self.conv1 = nn.Sequential(
            # input number of image channel
            nn.Conv2d(input_channel, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 12, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 12),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Linear(nf * 12 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, self.nf * 12 * 4 * 4)
        x = self.fc(x)
        return torch.sigmoid(x)


# Create the Discriminator
nf = 32
netD = Discriminator(nf).to(device)
# apply weight
netD.apply(weights_init)
print(netD)


##############################################
##  Generator
##############################################
class Generator(nn.Module):
    def __init__(self, z_size, nf, input_channel=3):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.nf = nf
        self.fc = nn.Linear(self.z_size, self.nf * 12 * 4 * 4)

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose2d(nf * 12, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True)
        )

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True)
        )

        self.dconv3 = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True)
        )

        self.dconv4 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True)
        )

        self.dconv5 = nn.Sequential(
            nn.ConvTranspose2d(nf, input_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.nf*12, 4, 4)
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        return self.dconv5(x)


# Create the generator
z_dim = 100
netG = Generator(z_dim, nf).to(device)
# apply weight
netG.apply(weights_init)
print(netG)


##############################################
##  Loss and optimizer
##############################################
# Initialize BCELoss function
criterion = nn.BCELoss()

# fixed noise for generator
fixed_noise = torch.randn(64, z_dim, device=device)

# Learning rate
lr = 0.0002
# Beta1 hyperparam
beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


##############################################
##  Training
##############################################
# Training
# Lists to keep track of progress
num_epochs = 2
img_list = []
G_losses = []
D_losses = []
iters = 0
step = 100


##############################################
##  Helper function for calculating loss
##############################################
def calculate_real_loss(x):
    batch = x.size(0)
    labels = torch.ones(batch, device=device)
    return criterion(x, labels)


def calculate_fake_loss(x):
    batch = x.size(0)
    labels = torch.ones(batch, device=device)
    return criterion(x, labels)


##############################################
##  Main training Function
##############################################
print("Training start...")
for epoch in range(num_epochs):
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        ############################
        # Train D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with real image
        optimizerD.zero_grad()
        # take images
        images = data[0].to(device)
        # batch size
        b_size = images.shape[0]
        # pass image to the ds model
        real_out = netD(images).view(-1)
        # Calculate loss
        loss_real = calculate_real_loss(real_out)
        loss_real.backward()
        dx = real_out.mean().item()

        ## Train with fake image
        # Generate noise
        noise = torch.randn(b_size, z_dim, device=device)
        # Generate fake image
        fake_image = netG(noise)
        # pass to ds model
        fake_out = netD(fake_image.detach()).view(-1)
        # calculate loss
        loss_fake = calculate_fake_loss(fake_out)
        loss_fake.backward()
        # calculate dz1
        dz1 = fake_out.mean().item()
        # calculate loss
        loss_D = loss_fake + loss_real
        # Update D
        optimizerD.step()

        ############################
        # Train G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        # crete new noise
        noise = torch.randn(b_size, z_dim, device=device)
        # Generate fake image
        fake_image = netG(noise)
        output = netD(fake_image).view(-1)
        # Calculate loss
        loss_G = calculate_real_loss(output)
        loss_G.backward()
        dz2 = output.mean().item()
        optimizerG.step()

        if i % step == 0:
            print('[%d/%d][%d/%d]\tD Loss: %.4f\tG Loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     loss_D.item(), loss_G.item(), dx, dz1, dz2))

        # Save Losses
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # update details if epoch is long
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(make_grid(fake, padding=2, normalize=True))

        iters += 1


##############################################
##  Compare Losses
##############################################
plt.figure(figsize=(10, 5))
plt.title("Generator vs Discriminator")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

##############################################
##  Compare Fake and Real Image
##############################################
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()


##############################################
##  Show a single Image
##############################################
def show_generated_img():
    noise = torch.randn(1, z_dim, 1, 1, device=device)
    gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    plt.imshow((gen_image + 1) / 2)
    plt.show()
