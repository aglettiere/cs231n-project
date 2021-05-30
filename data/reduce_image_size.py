from PIL import Image
import os

directory = '/Users/allisonlettiere/Downloads/cs231n-project/data/text_images_transparent/covers_png_only_text'
for filename in os.listdir(directory):
	filepath = os.path.join(directory, filename)
	image = Image.open(filepath)
	if len(image.split()) != 4:
		print(filepath)