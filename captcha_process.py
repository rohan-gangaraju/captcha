import numpy as np
import cv2

from os import listdir, makedirs
from os.path import join,isfile,exists

import csv


def processAllImages() :
	classified_folder = 'classified\\divided'
	csv_file = 'processed_input.csv'
	with open(csv_file, 'a') as captchaFile:
		writer = csv.writer(captchaFile);
		for dir in listdir(classified_folder) :
			captcha_class = dir
			class_folder = join(classified_folder, captcha_class)
			for file in listdir(class_folder) :
				full_path = join(class_folder, file)
				img = cv2.imread(full_path,0)
				dst = cv2.fastNlMeansDenoising(img, None, 45, 7, 21)
				thresh, im_bw = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
				#cv2.imshow("1",im_bw)
				bw_array = im_bw.flatten()
				#cv2.waitKey()
				bw_array[bw_array > 250] = 1
				print(bw_array)
				data = np.append(ord(dir),bw_array)
				print(data)
				writer.writerow(data)
				#input()

def processAllImagesCluster() :
	classified_folder = 'classified\\dividedCluster'
	csv_file = 'processed_input_cluster.csv'
	with open(csv_file, 'a') as captchaFile:
		writer = csv.writer(captchaFile);
		for dir in listdir(classified_folder) :
			captcha_class = dir
			class_folder = join(classified_folder, captcha_class)
			for file in listdir(class_folder) :
				full_path = join(class_folder, file)
				img = cv2.imread(full_path,0)
				bw_array = img.flatten()
				#cv2.imshow("some", img)
				#cv2.waitKey()
				bw_array[bw_array > 0] = 1
				print(bw_array)
				data = np.append(ord(dir),bw_array)
				print(data)
				writer.writerow(data)
				#input()

#processAllImages()
processAllImagesCluster()