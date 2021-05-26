import cv2
import pandas as pd
from ast import literal_eval
import csv
import matplotlib.pyplot as plt


def saveCoversAsPNG(data):
	for index, row in data.iterrows():
		image_path = row.image_path
		folders = image_path.split("/")
		if folders[-1] == 'cover_02_1964.jpg':
			continue
		image_path = '../' + '/'.join(folders[-3:])
		ocr_results = literal_eval(row.text_bounds)
		img = cv2.imread(image_path)
		covername = folders[-1].split('.')[0]
		print(image_path)
		# year = int(folders[-1].split('.')[0].split("_")[2])
		# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# plt.imshow(RGB_img)
		# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		# plt.show()
		# for line_text in ocr_results:
		# 	img = blur_line(line_text, img)
		# for line_text in ocr_results:
		# 	img = blur_line(line_text, img)
		cv2.imwrite("../data/covers_png/" + str(covername) + ".png", img)



def get_bounds(line_text):
	words = line_text["Words"]
	# print(line_text)
	max_height = int(line_text["MaxHeight"])
	min_top = int(line_text["MinTop"])
	start_idx_left = int(words[0]["Left"])
	end_idx_right = int(words[-1]["Left"] + words[-1]["Width"])
	start_idx_top = min_top
	end_idx_bottom = min_top + max_height
	# width = end_idx - start_idx

	# img = blur_word_left_right(min_top, start_idx, width, max_height, img)
	return [start_idx_top, end_idx_bottom, start_idx_left, end_idx_right]
def makePNGsTranspExceptText(data):
	dir_name = "../data/covers_png/"
	# pathlist = Path(dir_name).rglob('*.png')
	# print(pathlist)
	# for path in pathlist:
	for index, row in data.iterrows():
		image_path = row.image_path
		folders = image_path.split("/")
		if folders[-1] == 'cover_02_1964.jpg':
			continue
		image_path = '../' + '/'.join(folders[-3:])
		covername = folders[-1].split('.')[0]
		ocr_results = literal_eval(row.text_bounds)
		png_image_path = dir_name + covername + '.png'
		print(png_image_path)
		img = cv2.imread(png_image_path, cv2.IMREAD_UNCHANGED)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
		# print(img.shape)
		# print(img[100][50])
		all_bounds = []
		# plt.imshow(img)
		# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		# plt.show()
		for line_text in ocr_results:
			bounds = get_bounds(line_text)
			all_bounds.append(bounds)

		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				in_text_bounds = False
				for bound in all_bounds:
					if in_text_bounds:
						continue
					if i >= bound[0] and i <= bound[1] and j >= bound[2] and j <= bound[3]:
						in_text_bounds = True
				# print(in_text_bounds)
				if not in_text_bounds:
					rgba = img[i, j, :]
					rgba[-1] = 0
					img[i, j, :] = rgba
		# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# print(img)
		# plt.imshow(img)
		# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		# plt.show()
		cv2.imwrite("../data/covers_png_only_text/" + str(covername) + ".png", img)







csv_path = 'cover_text.csv'
data = pd.read_csv(csv_path)
# saveCoversAsPNG(data)
makePNGsTranspExceptText(data)

		


		
