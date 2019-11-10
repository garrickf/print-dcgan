import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=15000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
#parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# TODO: figure out discriminator, saving state, processing images, how to train

# G(z)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__() # Default constructor

        self.init_size = opt.img_size // 4
        # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        ## Added by me!
        self.kernel_size = 4
        self.ngf = 160 # Can make this editable
        self.z_size = 100
        self.out_size = 3

        """
        There are two alternatives to performing the hidden layer:
        transposed convolution/strided convolution, or convolution + upsampling.
        They are not the same...we stick to the original fomulation of strided convolution.
        
        ngf stands for the number of convolutional filters...
        """
        self.conv_blocks = nn.Sequential(
            # TODO: 4 4x4 transposed convolutional layers
            # so, kernel size = 4!
            # Layer 1 ()
            # ngf * 8 = 128 * 8 = 1024
            nn.ConvTranspose2d(self.z_size, self.ngf * 8, self.kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),
            # state size: (ngf * 8) x 4 x 4 -> 1024 x 4 x 4
            
            # Layer 2 ()
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),
            # state size: (ngf * 4) x 8 x 8
            
            # Layer 3
            # nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            # state size: (ngf * 2) x 16 x 16
            
            # Layer 4
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, self.kernel_size, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),
            # state size: ngf x 32 x 32

            # Output layer
            # nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.ConvTranspose2d(self.ngf, self.out_size, self.kernel_size, 2, 1, bias=False),
            nn.Tanh()
            # state size: out_size x 64 x 64

            # Added for ref: 
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        )

# Generator(
#   (conv_blocks): Sequential(
#     (0): ConvTranspose2d(100, 1024, kernel_size=(4, 4), stride=(1, 1), bias=False)
#     (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace)
#     (3): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU(inplace)
#     (6): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (7): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
#     (8): ReLU(inplace)
#     (9): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (11): ReLU(inplace)
#     (12): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
#     (13): Tanh()
#   )
# )

# -- input is Z, going into a convolution
# netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
# netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
# -- state size: (ngf*8) x 4 x 4
# netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
# netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
# -- state size: (ngf*4) x 8 x 8
# netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
# netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
# -- state size: (ngf*2) x 16 x 16
# netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
# netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
# -- state size: (ngf) x 32 x 32
# netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
# netG:add(nn.Tanh())
# -- state size: (nc) x 64 x 64

    """
    Takes in a Variable() as input and produces a Variable() as output
    """
    def forward(self, z):
        # out = self.l1(z)

        # Broadcast to a different shape
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(z)
        return img

# D
class Discriminator(nn.Module):
    '''
        Discriminative Network
    '''
    def __init__(self, in_size=3, ndf=40):
        super(Discriminator, self).__init__()
        self.in_size = in_size
        self.ndf = ndf

        self.main = nn.Sequential(
            # input size is in_size x 64 x 64
            nn.Conv2d(self.in_size, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size: 1 x 1 x 1
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        output = self.main(input)
        return output

# netG = Generator()
# print(netG)
# netD = Discriminator()
# print(netD)

def train_net(G, D, args, config):
    # cudnn.benchmark = True
    traindir = args.train_dir

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if config.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(traindir, True,
                    transforms.Compose([transforms.Scale(config.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]), download=True),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    elif config.dataset == 'celebA':
        train_loader = torch.utils.data.DataLoader(
                MydataFolder(traindir,
                    transform=transforms.Compose([transforms.Scale(config.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ])),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    else:
        return

    # setup loss function
    criterion = nn.BCELoss().cuda() 

    # setup optimizer
    optimizerD = torch.optim.Adam(D.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))

    # setup some varibles
    batch_time = AverageMeter()
    data_time = AverageMeter()
    D_losses = AverageMeter()
    G_losses = AverageMeter()

    fixed_noise = torch.FloatTensor(8 * 8, config.z_size, 1, 1).normal_(0, 1)
    fixed_noise = Variable(fixed_noise.cuda(), volatile=True)

    end = time.time()

    D.train()
    G.train()

    D_loss_list = []
    G_loss_list = []

    for epoch in range(config.epoches):
        for i, (input, _) in enumerate(train_loader):
            '''
                Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            '''
            data_time.update(time.time() - end)

            batch_size = input.size(0)
            input_var = Variable(input.cuda())

            # Train discriminator with real data
            label_real = torch.ones(batch_size)
            label_real_var = Variable(label_real.cuda())

            D_real_result = D(input_var).squeeze()
            D_real_loss = criterion(D_real_result, label_real_var)

            # Train discriminator with fake data
            label_fake = torch.zeros(batch_size)
            label_fake_var = Variable(label_fake.cuda())

            noise = torch.randn((batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
            noise_var = Variable(noise.cuda())
            G_result = G(noise_var)

            D_fake_result = D(G_result).squeeze()
            D_fake_loss = criterion(D_fake_result, label_fake_var)

            # Back propagation
            D_train_loss = D_real_loss + D_fake_loss
            D_losses.update(D_train_loss.data[0])

            D.zero_grad()
            D_train_loss.backward()
            optimizerD.step()

            '''
                Update G network: maximize log(D(G(z)))
            '''
            noise = torch.randn((batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
            noise_var = Variable(noise.cuda())
            G_result = G(noise_var)

            D_fake_result = D(G_result).squeeze()
            G_train_loss = criterion(D_fake_result, label_real_var)
            G_losses.update(G_train_loss.data[0])

            # Back propagation
            D.zero_grad()
            G.zero_grad()
            G_train_loss.backward()
            optimizerG.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.display == 0:
                print_log(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr,
                          config.display, batch_time, data_time, D_losses, G_losses)
                batch_time.reset()
                data_time.reset()
            elif (i + 1) == len(train_loader):
                print_log(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr,
                          (i + 1) % config.display, batch_time, data_time, D_losses, G_losses)
                batch_time.reset()
                data_time.reset()

        D_loss_list.append(D_losses.avg)
        G_loss_list.append(G_losses.avg)
        D_losses.reset()
        G_losses.reset()

        # plt the generate images and loss curve
        plot_result(G, fixed_noise, config.image_size, epoch + 1, args.save_dir, is_gray=(config.channel_size == 1))
        plot_loss(D_loss_list, G_loss_list, epoch + 1, config.epoches, args.save_dir)
        # save the D and G.
        save_checkpoint({'epoch': epoch, 'state_dict': D.state_dict(),}, os.path.join(args.save_dir, 'D_epoch_{}'.format(epoch)))
        save_checkpoint({'epoch': epoch, 'state_dict': G.state_dict(),}, os.path.join(args.save_dir, 'G_epoch_{}'.format(epoch)))

    #create_gif(config.epoches, args.save_dir)

# train_net(G, D, opt)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Loss function
adversarial_loss = torch.nn.BCELoss() # Idea: torch.nn.BCEWithLogitsLoss https://github.com/soumith/ganhacks/issues/36

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# class CustomDatasetFromFolder(data.Dataset):
#     def __init__(self, path):
#         images = glob.glob(in_folder + '*.jpg')
#         self.labels = labels
#         self.list_IDs = list_IDs

#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]

#         # Load data and get label
#         X = torch.load('data/' + ID + '.pt')
#         y = self.labels[ID]

#         return X, y

data_train = datasets.ImageFolder('./data2/', transform = transforms.Compose([
    # Can add other transformations in this list
    transforms.ToTensor()
]))
print(data_train)

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    data_train,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

os.makedirs("images", exist_ok=True)

Tensor = torch.FloatTensor

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        print (opt.batch_size)
        print (imgs.shape)
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False) # TODO: turn 1 and 0 into constants?
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # print(z.shape)
        z = Variable(torch.randn((opt.batch_size, 100)).view(-1, 100, 1, 1))
        print(z.shape)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True) # TODO: make images from same/deterministic noise sample to give better estimation/gif!