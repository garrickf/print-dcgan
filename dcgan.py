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

import logger # Our logging utility

import signal

"""
dcgan
---
Implements DCGAN for image generation.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=15000, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch size (drops last batch if too small)")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--z_size", type=int, default=100, help="size of the noise")
parser.add_argument("--img_size", type=int, default=64, help="size of image dimensions (square)")
parser.add_argument("--channels", type=int, default=3, help="number of image channels (default: RGB)")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between taking a snapshot of the images")
parser.add_argument("--experiment_name", type=str, default='experimentdefault', help="name of experiment")
parser.add_argument("--generate", default=False, action='store_true', help="set if generating images")
parser.add_argument("--resume", default=False, action='store_true', help="set if resuming training")
opt = parser.parse_args()

"""
Class: Generator/G(z)
---
Presents the implementation for the generator. Extends nn.Module.
"""
class Generator(nn.Module):
	def __init__(self, z_size=100, ngf=160, out_size=3):
		super(Generator, self).__init__() # Default constructor
		self.kernel_size = 4
		self.z_size = z_size
		self.ngf = ngf
		self.out_size = out_size

		"""
		There are two alternatives to performing the hidden layer I saw:
		transposed convolution/strided convolution, or convolution + upsampling.
		They are not the same; so, we stick to the original fomulation of strided convolution.
		
		Note: ngf stands for the number of convolutional filters.

		Note: Here's the interface for torch.nn.ConvTranspose2d:
			torch.nn.ConvTranspose2d(
				in_channels, out_channels, 
				kernel_size, stride=1, 
				padding=0, output_padding=0, 
				groups=1, bias=True, dilation=1, 
				padding_mode='zeros'
			)
		"""
		self.generate_image = nn.Sequential(
			# Layer 1 (100 x 1 noise to (ngf * 8) x 4 x 4)
			nn.ConvTranspose2d(self.z_size, self.ngf * 8, self.kernel_size, 1, 0, bias=False),
			nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(inplace=True),
			
			# Layer 2 ((ngf * 8) x 4 x 4 to (ngf * 4) x 8 x 8)
			nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, self.kernel_size, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(inplace=True),
			
			# Layer 3 ((ngf * 4) x 8 x 8 to (ngf * 2) x 16 x 16)
			nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, self.kernel_size, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 2, 0.8),
			nn.ReLU(inplace=True),
			
			# Layer 4 ((ngf * 2) x 16 x 16 to ngf x 32 x 32)
			nn.ConvTranspose2d(self.ngf * 2, self.ngf, self.kernel_size, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(inplace=True),

			# Output layer (ngf x 32 x 32 to out_size x 64 x 64)
			nn.ConvTranspose2d(self.ngf, self.out_size, self.kernel_size, 2, 1, bias=False),
			nn.Tanh()
		)

	"""
	Takes in a Variable() z of noise as input and produces a Variable() as output
	"""
	def forward(self, z):
		img = self.generate_image(z)
		return img

"""
Class: Discriminator/D(z)
---
Presents the implementation for the generator.
"""
class Discriminator(nn.Module):
	def __init__(self, in_size=3, ndf=35):
		super(Discriminator, self).__init__()
		self.in_size = in_size
		self.ndf = ndf

		"""
		Note: In soumith's version, they omit batch norm in the first layer as well.
		"""
		self.classify_image = nn.Sequential(
			# Layer 1 (in_size x 64 x 64 to ndf x 32 x 32)
			nn.Conv2d(self.in_size, self.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			
			# Layer 2 (ndf x 32 x 32 to (ndf * 2) x 16 x 16)
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),

			# Layer 3 ((ndf * 2) x 16 x 16 to (ndf * 4) x 16 x 16)
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),

			# Layer 4 ((ndf * 4) x 16 x 16 to (ndf * 8) x 4 x 4)
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			
			# Output layer state size: ((ndf * 8) x 4 x 4 to 1 x 1 x 1)
			nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid(),
		)

	def forward(self, input):
		output = self.classify_image(input)
		return output.view(-1, 1) # Convert to shape: [1, 1]

images_dir = opt.experiment_name + '/images/'
weights_dir = opt.experiment_name + '/weights/'

# Make directories
os.makedirs(images_dir, exist_ok=True) # Create directory to dump images/data
os.makedirs(weights_dir, exist_ok=True) # Create directory to dump weights into

"""
Generates a one-off image/set of images, given the experiment name. Uses the provided weights inside
the experiment folder to create a new Generator and, generate images from a noise sample.

Careful to run once and rename the image (./expname/images/generated.png) created, or else it will
be overwritten.

Example usage: 
	python dcgan.py --experiment_name <Experiment Name> --generate
	python dcgan.py --experiment_name experiment5 --batch_size 1
	python dcgan.py --experiment_name experiment7 --batch_size 1 --resume

"""
if opt.generate:
	print('Generating image...')
	generator = Generator()
	generator.load_state_dict(torch.load(weights_dir + 'G.tar')['weights'])
	fixed_noise = Variable(torch.randn((opt.batch_size, 100)).view(-1, 100, 1, 1))
	fixed_imgs = generator(fixed_noise)
	save_image(fixed_imgs.data[:25], opt.experiment_name + "/images/generated.png", nrow=5, normalize=True)
	print('All done!')
	exit()

"""
Otherwise, begin a new experiment!
"""
logger.setup_loggers(opt.experiment_name)
logger.log_debug('++ Experiment ' + opt.experiment_name + ' ++\n')
logger.log_debug('Arguments: %s\n' % opt)

# Helper to store state of D and G
def store_state(state, filename):
	torch.save(state, filename + '.tar')

# # Howto: Print model's state_dict
# print("G's state_dict:")
# for param_tensor in netG.state_dict():
#     print(param_tensor, "\t", netG.state_dict()[param_tensor].size())

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)

# Loss function
adversarial_loss = torch.nn.BCELoss() # Idea: torch.nn.BCEWithLogitsLoss https://github.com/soumith/ganhacks/issues/36

# Init generator, discriminator, optimizers
generator = Generator()
discriminator = Discriminator()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.FloatTensor

start_epoch = 0

if opt.resume: # Resume training from a checkpoint
	logger.log_debug('Attempting to resume training...')
	G_checkpoint = torch.load(weights_dir + 'G.tar')
	D_checkpoint = torch.load(weights_dir + 'D.tar')
	opt_checkpoint = torch.load(weights_dir + 'optimizer.tar')

	generator.load_state_dict(G_checkpoint['weights'])
	discriminator.load_state_dict(D_checkpoint['weights'])
	optimizer_G.load_state_dict(opt_checkpoint['optimizer_G'])
	optimizer_D.load_state_dict(opt_checkpoint['optimizer_D'])

	start_epoch = opt_checkpoint['epoch'] + 1 # Go one up
	logger.log_debug('Sucessfully loaded for epoch %s\n' % start_epoch)
else:
	logger.log_debug('Generator: %s\n' % generator)
	logger.log_debug('Discriminator: %s\n' % discriminator)

	# Init weights
	generator.apply(weights_init_normal)
	discriminator.apply(weights_init_normal)

# Define dataset. Put folders into /train_data/ to train on those.
data_train = datasets.ImageFolder('./train_data/', transform = transforms.Compose([
	# Can add other transformations in this list...
	transforms.ToTensor()
]))
logger.log_debug('Dataset: %s\n' % data_train)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
	data_train,
	batch_size=opt.batch_size,
	shuffle=True,
	drop_last=True # Drop last batch if too small
)

if opt.batch_size > len(data_train):
	logger.log_debug('ISSUE: batch size is too large; either decrease batch size or increase training data size.')
	quit()

"""
Training
"""
def signal_handler(sig, frame):
	logger.log_debug(' Force quit, saving and exiting program...')
	store_state({'epoch': epoch, 'weights': discriminator.state_dict()}, 
		os.path.join(weights_dir, 'D'))
	store_state({'epoch': epoch, 'weights': generator.state_dict()}, 
		os.path.join(weights_dir, 'G'))
	store_state({'epoch': epoch, 'optimizer_G': optimizer_G.state_dict(), 'optimizer_D': optimizer_D.state_dict()},
		os.path.join(weights_dir, 'optimizer'))
	exit()
signal.signal(signal.SIGINT, signal_handler)

fixed_noise = Variable(torch.randn((opt.batch_size, 100)).view(-1, 100, 1, 1)) # Use for visualizations
intervals_done = int(start_epoch * len(dataloader) // opt.sample_interval) # Epoch * num_batches / sample_interval
# print(intervals_done)

for epoch in range(start_epoch, opt.n_epochs):
	for i, (imgs, _) in enumerate(dataloader):
		# Valid images are labelled y=1
		valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

		# Configure input
		real_imgs = Variable(imgs.type(Tensor))

		####### Train generator
		optimizer_G.zero_grad()

		# Sample noise for input to generator
		z = Variable(torch.randn((opt.batch_size, 100)).view(-1, 100, 1, 1))

		# Generate a batch of images
		gen_imgs = generator(z)

		# print(discriminator(gen_imgs).size())
		# print(valid.size())
		# Loss measures generator's ability to fool the discriminator
		g_loss = adversarial_loss(discriminator(gen_imgs), valid)
		g_loss.backward()
		optimizer_G.step()

		####### Train discriminator
		optimizer_D.zero_grad()

		# Loss describes discriminator's ability to classify real from generated samples, so we use both
		real_loss = adversarial_loss(discriminator(real_imgs), valid)
		fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
		d_loss = (real_loss + fake_loss) / 2

		d_loss.backward()
		optimizer_D.step()

		batches_done = epoch * len(dataloader) + i

		logger.log_debug(
			'[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]'
			% (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
		)

		# Log data to CSV file
		logger.log_data(
			iteration=batches_done, 
			dloss=d_loss.item(), 
			gloss=g_loss.item(),
			epoch=epoch, 
			batch=i)

		# Every sampling interval, we save an image of our progress so far
		if batches_done % opt.sample_interval == 0:
			fixed_imgs = generator(fixed_noise)
			save_image(fixed_imgs.data[:25], opt.experiment_name + "/images/seq%d.png" % intervals_done, nrow=5, normalize=True)
			intervals_done += 1

	# Every epoch, save the weights and optimizer state
	# https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/2
	print('Saving weights...')
	store_state({'epoch': epoch, 'weights': discriminator.state_dict()}, 
		os.path.join(weights_dir, 'D'))
	store_state({'epoch': epoch, 'weights': generator.state_dict()}, 
		os.path.join(weights_dir, 'G'))
	store_state({'epoch': epoch, 'optimizer_G': optimizer_G.state_dict(), 'optimizer_D': optimizer_D.state_dict()},
		os.path.join(weights_dir, 'optimizer'))
