import cv2
import pandas as pd
from ast import literal_eval
import csv
import matplotlib.pyplot as plt
import os
from pathlib import Path
import kornia
import torch

def blurImages(dir_name, new_dirname):
	
	# directory = os.fsencode(dir_name)
	pathlist = Path(dir_name).rglob('*.jpg')
	print(pathlist)
	for path in pathlist:
		print(path)
		img: np.ndarray = cv2.imread(str(path))
		cover_name = str(path).split("/")[-1]
		# convert to torch tensor
		data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW

		# create the operator
		gauss = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))

		# blur the image
		x_blur: torch.tensor = gauss(data.float())

		# convert back to numpy
		img_blur: np.ndarray = kornia.tensor_to_image(x_blur.byte())
		newpath = new_dirname + cover_name
		print(newpath)
		cv2.imwrite(newpath, img_blur)
		# Create the plot
		# fig, axs = plt.subplots(1, 2, figsize=(16, 10))
		# axs = axs.ravel()

		# axs[0].axis('off')
		# axs[0].set_title('image source')
		# axs[0].imshow(img)

		# axs[1].axis('off')
		# axs[1].set_title('image blurred')
		# axs[1].imshow(img_blur)
		# plt.show()


dir_name = '../data/blurred_images/'
new_dirname = "../data/no_text_and_gauss_blur/"
blurImages(dir_name, new_dirname)