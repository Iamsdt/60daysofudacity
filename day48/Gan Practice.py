# %matplotlib inline
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

# Set random seem for reproducibility
manualSeed = 963
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Hyper parameters

# Number of workers
num_workers = 4

# Batch size
batch_size = 128

# image size
image_size = 64

# num of image channel
nc = 3

# Size of z latent vector (generator input)
nz = 100

# feature maps number in generator
ngf = 64

# feature maps number in discriminator
ndf = 64

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# custom weights initialization for models
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # num of z input
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # second layer
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # third layer
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # fourth layer
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # last layer
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# Create the generator
netG = Generator().to(device)
# apply weight
netG.apply(weights_init)
print(netG)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input number of image channel
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # second layer
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # third layer
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # fourth layer
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # last layer
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Create the Discriminator
netD = Discriminator().to(device)
# apply weight
netD.apply(weights_init)
print(netD)

path = '../input/samples/'
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

# Visualize
images, _ = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
grid = make_grid(images.to(device)[:64], padding=2, normalize=True).cpu()
img = np.transpose(grid, (1, 2, 0))
plt.imshow(img)

# Initialize BCELoss function
criterion = nn.BCELoss()

# fixed noise for generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# real label and fake label
real_label = 1
fake_label = 0

# Learning rate
lr = 0.0002
# Beta1 hyperparam
beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training
# Lists to keep track of progress
num_epochs = 10
img_list = []
G_losses = []
D_losses = []
iters = 0
step = 100

print("Training start...")
for epoch in range(num_epochs):
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        ############################
        # Train D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with real image
        netD.zero_grad()
        # take images and labels
        images = data[0].to(device)
        # take batch size
        b_size = images.size(0)
        # create real label
        labels = torch.full((b_size,), real_label, device=device)
        # pass image to the ds model
        real_out = netD(images).view(-1)
        # Calculate loss
        loss_real = criterion(real_out, labels)
        loss_real.backward()
        dx = real_out.mean().item()

        ## Train with fake image
        # Generate noise
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image
        fake_image = netG(noise)
        # fill label with fake value: 0
        labels.fill_(fake_label)
        # pass to ds model
        fake_out = netD(fake_image.detach()).view(-1)
        # calculate loss
        loss_fake = criterion(fake_out, labels)
        loss_fake.backward()
        # calculate dz1
        dz1 = output.mean().item()
        # calculate loss
        loss_D = loss_fake + loss_real
        # Update D
        optimizerD.step()

        ############################
        # Train G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # fill with real labels
        labels.fill_(real_label)
        # pass fake image previously created
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image
        fake_image = netG(noise)
        output = netD(fake_image).view(-1)
        # Calculate loss
        loss_G = criterion(output, labels)
        loss_G.backward()
        dz2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % step == 0:
            print('[%d/%d][%d/%d]\tD Loss: %.4f\tG Loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     loss_D.item(), loss_G.item(), dx, dz1, dz2))

        # Save Losses for plotting later
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(make_grid(fake, padding=2, normalize=True))

        iters += 1

# Compare losses
plt.figure(figsize=(10, 5))
plt.title("Generator vs Discriminator")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Compare outputs
# Grab a batch of real images from the dataloader
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


# show single image
def show_generated_img():
    noise = torch.randn(1, nz, 1, 1, device=device)
    gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    plt.imshow((gen_image + 1) / 2)
    plt.show()
