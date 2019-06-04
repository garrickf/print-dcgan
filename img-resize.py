# For whateven reason, this links to my installation of python2.
# Might need to check pip and conda

import os
import cv2
import glob

# Change input/output
in_folder = './pop-art-original-images/' 
out_folder = './pop-art-resized-images/'

images = glob.glob(in_folder + '*.jpg')
print('Found files, two sample file names below:')
print(images[:2])

width = 64
height = 64

print('Resizing images to be %d x %d...' % (width, height))

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

for path in images:
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	img_height, img_width = img.shape[:2]
	if img_width < img_height:
		# Width is limiting, scale down and crop
		adj_height = round(img_height * width / img_width)
		img = cv2.resize(img, (width, adj_height))
		y_start = round(adj_height / 2 - height / 2)
		img = img[y_start:y_start+height, :]
	else:
		adj_width = round(img_width * height / img_height)
		img = cv2.resize(img, (adj_width, height))
		x_start = round(adj_width / 2 - width / 2)
		img = img[:, x_start:x_start+width]
	cv2.imwrite(path.replace(in_folder, out_folder), img)
