from PIL import Image
import os.path

cover_art_string = "cover_art_epoch_"
font_string = "fonts_epoch_"

directory_path = "/Users/allisonlettiere/Downloads/cs231n-project/src/lbfgs2/"

for epoch in range(0,100):
	for iters in range(99):
		cover_filename = cover_art_string + str(epoch) + "_" + str(iters) + ".png"
		font_filename = font_string + str(epoch) + "_" + str(iters) + ".png"
		if os.path.isfile(directory_path + cover_filename) and os.path.isfile(directory_path + font_filename):
			cover_image = Image.open(directory_path + cover_filename).convert("RGBA")
			font_image = Image.open(directory_path + font_filename).convert("RGBA")
			
			'''datas = font_image.getdata()

			newData = []
			for item in datas:
				if item[3] != 0:
					newData.append((255, 255, 255, 255))
				else:
					newData.append(item)

			font_image.putdata(newData)
			font_image.show()'''

			cover_image.paste(font_image, (0, 0), font_image)

			cover_image.save("/Users/allisonlettiere/Downloads/cs231n-project/results/lbfgs2-results-combined/epoch_"+ str(epoch) + "_" + str(iters) + ".png")