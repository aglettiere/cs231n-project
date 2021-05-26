from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from PIL import Image


from scipy.optimize import differential_evolution

# Set random seed for reproducibility #
#manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot_cover_art = '/Users/allisonlettiere/Downloads/cs231n-project/data/cover_art_images_blurred/'
dataroot_fonts = '/Users/allisonlettiere/Downloads/cs231n-project/data/text_images_transparent/'
dataroot_full_covers = '/Users/allisonlettiere/Downloads/cs231n-project/data/full_covers/'
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 4

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128

# Number of training epochs
num_epochs = 1

# Learning rate for optimizers -- try hyperparameter search, learning rate decay
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset_cover_art = dset.ImageFolder(root=dataroot_cover_art,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]),
                           loader=custom_loader)

dataset_fonts = dset.ImageFolder(root=dataroot_fonts,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]),
                           loader=custom_loader)

dataset_full_covers = dset.ImageFolder(root=dataroot_cover_art,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloaders
dataloader_cover_art = torch.utils.data.DataLoader(dataset_cover_art, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

dataloader_fonts = torch.utils.data.DataLoader(dataset_fonts, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

'''dataloader_full_covers = torch.utils.data.DataLoader(dataset_full_covers, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)'''

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the generators
netG_cover_art = Generator(ngpu).to(device)
netG_fonts = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG_cover_art = nn.DataParallel(netG_cover_art, list(range(ngpu)))
    netG_fonts = nn.DataParallel(netG_fonts, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG_cover_art.apply(weights_init)
netG_fonts.apply(weights_init)

# Print the models
print(netG_cover_art)
print(netG_fonts)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminators
netD_cover_art = Discriminator(ngpu).to(device)
netD_fonts = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD_cover_art = nn.DataParallel(netD_cover_art, list(range(ngpu)))
    netD_fonts = nn.DataParallel(netD_fonts, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD_cover_art.apply(weights_init)
netD_fonts.apply(weights_init)

# Print the model
print(netD_cover_art)
print(netD_fonts)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D --- try L-BFGS
optimizerG_cover_art = optim.Adam(netG_cover_art.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG_fonts = optim.Adam(netG_fonts.parameters(), lr=lr, betas=(beta1, 0.999))

optimizerD_cover_art = optim.Adam(netD_cover_art.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD_fonts = optim.Adam(netD_fonts.parameters(), lr=lr, betas=(beta1, 0.999))

'''
optimizerG_cover_art = optim.LBFGS(netG_cover_art.parameters(), lr=0.1, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
optimizerG_fonts = optim.LBFGS(netG_fonts.parameters(), lr=0.1, max_iter=20, history_size=100, line_search_fn='strong_wolfe')

optimizerD_cover_art = optim.LBFGS(netD_cover_art.parameters(), lr=0.1, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
optimizerD_fonts = optim.LBFGS(netD_fonts.parameters(), lr=0.1, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
'''

'''
optimizerG_cover_art = torch.optim.RMSprop(netG_cover_art.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0.001, momentum=0)
optimizerG_fonts = torch.optim.RMSprop(netG_fonts.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0.001, momentum=0)

optimizerD_cover_art = torch.optim.RMSprop(netD_cover_art.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0.001, momentum=0)
optimizerD_fonts = torch.optim.RMSprop(netD_fonts.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0.001, momentum=0)
'''

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * (0.1 ** (epoch // 30))

# Training Loop

# Lists to keep track of progress
cover_art_img_list = []
font_img_list = []

image_dict = {}
full_img_list = []

G_cover_art_losses = []
G_font_losses = []
D_cover_art_losses = []
D_font_losses = []
iters = 0

print("Starting Training Loop...")

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    adjust_learning_rate(optimizerG_cover_art, epoch)
    adjust_learning_rate(optimizerG_fonts, epoch)
    adjust_learning_rate(optimizerD_cover_art, epoch)
    adjust_learning_rate(optimizerD_fonts, epoch)

    for i, data in enumerate(zip(dataloader_fonts, dataloader_cover_art)):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD_cover_art.zero_grad()
        # Format batch
        real_cpu = data[0][0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD_cover_art(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_covert_art_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_covert_art_real.backward()
        D_cover_art_x = output.mean().item()

        netD_fonts.zero_grad()
        # Format batch
        real_cpu = data[1][0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD_fonts(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_fonts_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_fonts_real.backward()
        D_fonts_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake_covers = netG_cover_art(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD_cover_art(fake_covers.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_cover_art_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_cover_art_fake.backward()
        D_G_z1_cover_art = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD_cover_art = errD_covert_art_real + errD_cover_art_fake
        # Update D
        #optimizerD_cover_art.step()

        # Genetic Algorithm Code -- update situation here
        obj = errD_cover_art_fake.item()
        bounds = [(-5, 5), (-5, 5)]
        print("OBJ")
        print(obj)
        print("Old parameters")
        print(netD_cover_art.parameters())
        new_params = differential_evolution(obj, bounds)
        '''
        class Objective(nn.Module):
            def __init__(self, loss, output, labels):
                super(Objective, self).__init__()
                self.loss = loss
                self.output = output
                self.labels = labels

                

            def forward(self):
                # output = self.linear(self.X)
                return self.loss(output, labels)


        objective = Objective(criterion, output, label)
        obj = PyTorchObjective(objective)
        xL = optimize.differential_evolution(obj.fun, bounds, callback=verbose)
        '''
        print("New parameters")
        print(new_params)
        with torch.no_grad():
            netD_cover_art.parameters().copy_(new_params)
        exit()
        
        # Generate fake image batch with G
        fake_fonts = netG_fonts(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD_fonts(fake_fonts.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fonts_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fonts_fake.backward()
        D_G_z1_fonts = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD_fonts = errD_fonts_real + errD_fonts_fake
        # Update D
        optimizerD_fonts.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG_cover_art.zero_grad()
        netG_fonts.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD_cover_art(fake_covers).view(-1)
        # Calculate G's loss based on this output
        errG_cover_art = criterion(output, label)
        # Calculate gradients for G
        errG_cover_art.backward()
        D_G_z2_cover_art = output.mean().item()
        # Update G
        optimizerG_cover_art.step()

        output = netD_fonts(fake_fonts).view(-1)
        # Calculate G's loss based on this output
        errG_fonts = criterion(output, label)
        # Calculate gradients for G
        errG_fonts.backward()
        D_G_z2_fonts = output.mean().item()
        # Update G
        optimizerG_fonts.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tCover Art Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader_fonts),
                     errD_cover_art.item(), errG_cover_art.item(), D_cover_art_x, D_G_z1_cover_art, D_G_z2_cover_art))
            print('[%d/%d][%d/%d]\tFont Images Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader_fonts),
                     errD_fonts.item(), errG_fonts.item(), D_fonts_x, D_G_z1_fonts, D_G_z2_fonts))

        # Save Losses for plotting later
        G_cover_art_losses.append(errG_cover_art.item())
        G_font_losses.append(errG_fonts.item())
        D_cover_art_losses.append(errD_cover_art.item())
        D_font_losses.append(errD_fonts.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 1 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader_fonts)-1)):
            with torch.no_grad():
                fake_cover_art = netG_cover_art(fixed_noise).detach().cpu()
                fake_fonts = netG_fonts(fixed_noise).detach().cpu()
            
            vutils.save_image(fake_cover_art, "cover_art_epoch_" +str(epoch)+".png")
            vutils.save_image(fake_fonts, "fonts_epoch_" +str(epoch)+".png")
            
            cover_art_img_list.append(vutils.make_grid(fake_cover_art, padding=2, normalize=True))
            font_img_list.append(vutils.make_grid(fake_fonts, padding=2, normalize=True))

            image_dict[epoch] = (fake_cover_art, fake_fonts)

        iters += 1

#torch.save(netG.state_dict(), "generator_mixed.pt")
#torch.save(netD.state_dict(), "discriminator_mixed.pt")

composite_iters = 0
for image_epoch in image_dict:
    for image_pair in zip(image_dict[image_epoch][0], image_dict[image_epoch][1]):
        fake_cover_art_image, fake_font_image = image_pair
        cover_art_pil = transforms.ToPILImage()(fake_cover_art_image.squeeze_(0)).convert('RGBA')
        font_img_pil = transforms.ToPILImage()(fake_font_image.squeeze_(0)).convert('RGBA')
        image_composite = Image.alpha_composite(cover_art_pil, font_img_pil)
        image_composite.save('composite_image_epoch_' + str(image_epoch) + '_image_' + str(composite_iters), 'PNG')
        composite_iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Losses During Training")
plt.plot(G_cover_art_losses,label="G Cover Art")
plt.plot(G_font_losses,label="G Font")
plt.plot(D_cover_art_losses,label="D Cover Art")
plt.plot(D_font_losses,label="D Font")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('LossPlotTwoGTwoD.png')

'''fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
plt.savefig('image_list.png')'''