import cv2
import pandas as pd
from ast import literal_eval
import csv
import matplotlib.pyplot as plt
def blur_word_left_right(dim_0_start_idx, dim_1_start_idx, word_width, word_height, img):
	filler_size = 20
	dim_0_end_idx = dim_0_start_idx + word_height
	dim_1_end_idx = dim_1_start_idx + word_width
	filler_left_start = dim_1_start_idx - (filler_size + 1)
	filler_left_end = dim_1_start_idx - 1
	filler_right_start = dim_1_end_idx
	filler_right_end = dim_1_end_idx + filler_size
	#left half
	half_width = int(word_width/2)
	increments = int(half_width/filler_size)
	left_start_idx = dim_1_start_idx
	left_end_idx = dim_1_start_idx + half_width
	right_start_idx = left_end_idx
	right_end_idx = dim_1_start_idx + word_width
	start_idx = left_start_idx
	while start_idx < left_end_idx:
		img[dim_0_start_idx:dim_0_end_idx, start_idx:start_idx + filler_size, :] = img[dim_0_start_idx:dim_0_end_idx, filler_left_start:filler_left_end, :]
		start_idx += filler_size
	start_idx = right_start_idx
	while start_idx < right_end_idx:
		img[dim_0_start_idx:dim_0_end_idx, start_idx:start_idx + filler_size, :] = img[dim_0_start_idx:dim_0_end_idx, filler_right_start:filler_right_end, :]
		start_idx += filler_size
	
	return img
def blur_line(line_text, img):
	words = line_text["Words"]
	# print(line_text)
	max_height = int(line_text["MaxHeight"])
	min_top = int(line_text["MinTop"])
	start_idx = int(words[0]["Left"])
	end_idx = int(words[-1]["Left"] + words[-1]["Width"])
	width = end_idx - start_idx
	img = blur_word_left_right(min_top, start_idx, width, max_height, img)
	return img

csv_path = 'cover_text.csv'
data = pd.read_csv(csv_path)
for index, row in data.iterrows():
	image_path = row.image_path
	folders = image_path.split("/")
	image_path = '../' + '/'.join(folders[-3:])
	ocr_results = literal_eval(row.text_bounds)
	img = cv2.imread(image_path)
	covername = folders[-1]
	if folders[-1] == 'cover_02_1964.jpg':
		continue
	print(image_path)
	year = int(folders[-1].split('.')[0].split("_")[2])
	# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# plt.imshow(RGB_img)
	# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	# plt.show()
	for line_text in ocr_results:
		# print(line_text)
		img = blur_line(line_text, img)
	for line_text in ocr_results:
		img = blur_line(line_text, img)
	cv2.imwrite("../data/blurred_images/" + str(covername), img)
	# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# plt.imshow(RGB_img)
	# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	# plt.show()
	
	