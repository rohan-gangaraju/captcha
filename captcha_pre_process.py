#Script which contains methods for pre-processing of captcha images such as fetching captcha images, saving the base64 encoded images to png files and classifying png files into different digits

import numpy as np
import cv2	

from os import listdir, makedirs
from os.path import join,isfile,exists
from shutil import rmtree

import base64

import urllib.request
from lxml import html
import requests
import csv

#np.set_printoptions(threshold=np.inf)

def getImgData() :
	page = requests.get("https://mailsrv.cs.umass.edu/webmail/");
	tree = html.fromstring(page.content)

	img = tree.xpath('//*[@id="login-form"]/div[1]/form/table[2]/tbody/tr/td[1]/img/@src');
	txt = tree.xpath('//*[@id="bottomline"]');
	#print(txt[0].text)

	#print (img[0])
	return img[0]

# Method to make multiple calls to the Umass site, fetch the base64 png data and write the data to a csv file
def writeCaptchaToCSV() :
	with open('pycaptcha.csv', 'a') as captchaFile:
		for i in range(0,5000) :
			print(i)
			writer = csv.writer(captchaFile);
			writer.writerow([getImgData()])
		
# Method to convert the base64 encodings into PNG's which can be later read as pixel data
def convertBase64ToPng() :
	with open('pycaptcha.csv', 'r') as captchaFile:
		i = 1
		for line in captchaFile:
			
			if line != "\n" :
				data = line.strip().split(',')
				#print(data[1])
				
				path = "images"
				pngfile = str(i) + ".png"
				file = join(path, pngfile)
				print(file)
				with open(file, "wb") as fh:
					fh.write(base64.b64decode(data[1]))
				i += 1
		

# Method to read entire captcha image, filter and split it into 4 parts and save the images under respective character folder based on the class.txt file which contains the classification details
def saveClassfiedImages() :
	# Path of classified images
	parent_path = 'classified'

	captcha_value_list = []

	with open(join(parent_path,"class.txt"),"r") as classFile:
		for line in classFile:
			captcha_value_list.append(line.strip())

	onlyfiles = [f for f in listdir(join(parent_path,'fullimages')) if isfile(join(parent_path,'fullimages', f))]

	print(onlyfiles)

	rows = 60
	cols = 180

	for file in onlyfiles:
		print("File -------------->" , file)
		full_path = join(parent_path,'fullimages',file)
		index, png = file.split('.')
		index = int(index) - 1
		img = cv2.imread(full_path,0)
		#cv2.imshow("some",img)
		print(img.shape)
		
		'''
		# Code to crop the image to rows,cols
		
		if img.shape[0] < rows :
			diff = rows - img.shape[0]
			zeros = np.zeros((diff,img.shape[1]),dtype=np.uint8)	# zeros uint8 is crucial, otherwise you will just see a black image in imshow . 
			img = np.r_[img,zeros]
			print(img[0])
			#cv2.imshow("i2222222",img)
		elif img.shape[0] > rows : 
			img = img[:rows]
			
		if img.shape[1] < cols :
			diff = cols - img.shape[1]
			zeros = np.zeros((img.shape[0],diff),dtype=np.uint8)
			img = np.c_[img,zeros]
			
		elif img.shape[1] > cols : 
			img = img[:,:cols]
			
		print(img.shape)
		print(img[0])
		'''
		img = cv2.resize(img, (cols,rows))
		
		x = np.array_split(img,4,1)
		print(x[0].shape)
		#cv2.imshow("12222",x[0])
		#cv2.imshow("2222222",x[1])
		#cv2.imshow("322222222",x[2])
		#cv2.imshow("422222",x[3])
		
		print(index)
		print(captcha_value_list[index][0], captcha_value_list[index][1], captcha_value_list[index][2], captcha_value_list[index][3])
		
		for i in range(0,4) :
			directory = join(parent_path,'divided',captcha_value_list[index][i])
			if not exists(directory):
				makedirs(directory)
				
			existingfiles = [f for f in listdir(directory) if isfile(join(directory,f))]
			imgfile = str(len(existingfiles)) + ".png"
			imgname = join(directory,imgfile)
	#		print(directory)
			print(imgname)
			cv2.imwrite(imgname, x[i] );
		#cv2.waitKey(2000)
		#cv2.destroyAllWindows()
		#input()

	
def splitImage(image) :

	clrImage = cv2.fastNlMeansDenoising(image, None, 45, 7, 21)
	
	#kernel1 = np.ones((2,2),np.uint8)
	#clrImage = cv2.erode(clrImage,kernel1,iterations = 1)
	
	kernel2 = np.ones((2,2),np.uint8)
	#dst = cv2.morphologyEx(cur_img, cv2.MORPH_OPEN, kernel)
	clrImage = cv2.dilate(clrImage,kernel2,iterations = 1)

	rows = 60
	cols = 180
	
	clrImage = cv2.resize(clrImage, (cols,rows))

	thresh, im_bw = cv2.threshold(clrImage, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # If OTSU's method is used, then the threshold value used is returned
	
	index_pairs = []
		
	for i in range(0,im_bw.shape[0]) :
		for j in range(0,im_bw.shape[1]) :
			if im_bw[i][j] > 0 :	#non-white pixels
				index_pairs.append(np.array([i,j]))

	index_pairs_array = np.array(index_pairs,dtype=np.float32)
	k = 4
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(index_pairs_array,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	image_array = []
	min_col_array = []
	
	for i in range(0,4) :
		dict = {0:None, 1:None}
		subImageCoord = index_pairs_array[label.ravel() == i]
		subrows,subcols = np.hsplit(subImageCoord,2)
		
		mincol = int(np.amin(subcols))
		maxcol = int(np.amax(subcols))
		totalcols = maxcol - mincol + 1
		
		subImage = np.zeros([rows,totalcols])
		
		for j in subImageCoord :
			r = int(j[0])
			c = int(j[1]) - mincol
			subImage[r][c] = 255

		subImage = cv2.resize(subImage, (int(cols/4),rows))
		
		# resize will introduce grayscale pixel values
		subImage[subImage > 0] = 255
		
		image_array.append(subImage)
		min_col_array.append(mincol)
		
	sorted_images = []
	
	sorted_index = np.argsort(min_col_array)	
	
	for i in sorted_index:
		sorted_images.append(image_array[i])
		#print(image_array[i].flatten())
		#cv2.imshow("show", image_array[i])
		#cv2.waitKey()
		
	return sorted_images	

def saveClassfiedImagesCluster() :
	# Path of classified images
	parent_path = 'classified'
	
	# Remove existing divided files
	rmtree(join(parent_path,'dividedCluster'))

	captcha_value_list = []

	with open(join(parent_path,"class.txt"),"r") as classFile:
		for line in classFile:
			captcha_value_list.append(line.strip())

	onlyfiles = [f for f in listdir(join(parent_path,'fullimages')) if isfile(join(parent_path,'fullimages', f))]

	#print(onlyfiles)

	rows = 60
	cols = 180

	for file in onlyfiles:
		print("File -------------->" , file)
		full_path = join(parent_path,'fullimages',file)
		index, png = file.split('.')
		index = int(index) - 1
		img = cv2.imread(full_path,0)
		
		split_images = splitImage(img)
		
		for i in range(0,4) :
			directory = join(parent_path,'dividedCluster',captcha_value_list[index][i])
			if not exists(directory):
				makedirs(directory)
				
			existingfiles = [f for f in listdir(directory) if isfile(join(directory,f))]
			imgfile = str(len(existingfiles)) + ".png"
			imgname = join(directory,imgfile)
	#		print(directory)
			print(imgname)
			cv2.imwrite(imgname, split_images[i] );
		#cv2.waitKey(2000)
		#cv2.destroyAllWindows()
		#input()

def run() :
	#writeCaptchaToCSV()
	#convertBase64ToPng()
	#saveClassfiedImages()
	saveClassfiedImagesCluster()
	
if __name__ == "__main__" :
	run()