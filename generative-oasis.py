"""
This code is based off the example provided by

https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import matplotlib.pyplot as plt
import numpy as np

# Configure the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA Not Found. Using CPU")

"""
=========================================================================================================================================
Setup Data Pipeline
=========================================================================================================================================
"""

trainRoot = '~/COMP3710/Pattern Recognition Demo/Generative/training'
testRoot = '~/COMP3710/Pattern Recognition Demo/Generative/testing'
workers = 1 # Number of workers for dataloader
batch_size = 64 # Batch size during training
image_size = 64 # Spatial size of training images. All images will be resized to this size using a transformer.
nc = 3 # Number of channels in the training images. For color images this is 3, greyscale 1
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
num_epochs = 10 # Number of training epochs
lr = 1e-4 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers

# Create the dataset
trainset = torchvision.datasets.ImageFolder(root=trainRoot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Create the dataset
testset = torchvision.datasets.ImageFolder(root=testRoot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
"""
=========================================================================================================================================
Model
=========================================================================================================================================
"""
# Custom weights initialization called on ``netG`` and ``netD``
# This generates noise from a normal distribution with mean 0 and std deviation 0.02
# These are the parameters specified by the DCGAN Paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Size is (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Size is (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Size is (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Size is (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Size is (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

#Discriminator Code
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size is (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size is (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size is (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size is (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator().to(device)
#Initialise all weights to be randomly distributed with mean 0 and std deviation 0.02
netG.apply(weights_init)
# Create the Discriminator
netD = Discriminator().to(device)
#Initialise all weights to be randomly distributed with mean 0 and std deviation 0.02
netD.apply(weights_init)
# Initialize the loss function
criterion = nn.BCELoss()
# Create batch of latent vectors
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

"""
=========================================================================================================================================
Train the Model
=========================================================================================================================================
"""
# Training Loop
img_list = []
G_losses = []
D_losses = []
iters = 0

startTime = time.time()
print("> Training")
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        """
        Train with all-real batch
        """ 
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        """
        Train with all-fake batch
        """ 
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # The fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Append losses to array for tracking
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


endTime = time.time()
runTime = endTime - startTime
print("Training Time: "+ str(runTime) +" seconds")

# Plot the generator and discriminator loss
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('generator_discriminator_loss.png')
plt.close()

"""
=========================================================================================================================================
Test the Model
=========================================================================================================================================
"""

print("> Testing")
# Grab a batch of real images from the dataloader
real_batch = next(iter(train_loader))

# Save plots
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))

plt.savefig('images.png')
plt.close()

# Testing Phase
print("Starting Testing Phase...")
netD.eval()  # Set Discriminator to evaluation mode

correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for data in test_loader:
        real_cpu = data[0].to(device)
        label = torch.full((real_cpu.size(0),), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        test_loss += criterion(output, label).item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = torch.full((images.size(0),), real_label, dtype=torch.float, device=device)
        outputs = netD(images).view(-1)
        predicted = (outputs > 0.5).float()  # Classify based on 0.5 threshold
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")