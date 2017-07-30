import numpy as np
import cv2

from os import listdir, makedirs
from os.path import join,isfile,exists

import csv
import neurallib as nl

import sys
#import pdb

import captcha_pre_process as cpp

numdict = { 
	50: [0,0,0,0,0], 
	51: [0,0,0,0,1], 
	52: [0,0,0,1,0], 
	53: [0,0,0,1,1], 
	54: [0,0,1,0,0], 
	55: [0,0,1,0,1], 
	56: [0,0,1,1,0], 
	57: [0,0,1,1,1],
	65: [0,1,0,0,0], 
	66: [0,1,0,0,1], 
	67: [0,1,0,1,0], 
	68: [0,1,0,1,1], 
	69: [0,1,1,0,0], 
	70: [0,1,1,0,1], 
	71: [0,1,1,1,0], 
	72: [0,1,1,1,1],	
	75: [1,0,0,0,0], 
	77: [1,0,0,0,1], 
	78: [1,0,0,1,0], 
	80: [1,0,0,1,1], 
	82: [1,0,1,0,0], 
	83: [1,0,1,0,1], 
	84: [1,0,1,1,0], 
	86: [1,0,1,1,1],
	87: [1,1,0,0,0], 
	88: [1,1,0,0,1], 
	89: [1,1,0,1,0], 
	90: [1,1,0,1,1]
	}
	
def readFromFile(file) :
	return np.genfromtxt(file, delimiter=",")
	
def processInput(narray) :
	nColumns = narray.shape[1]
	inputY, inputX = np.split(narray,[1], axis=1)

	modifiedY = []

	for i in range(0,len(inputY)) : 
		character = chr(inputY[i]);
		print("char : ", character)
		
		class_img_path = "classified\\image_classes"
		img = cv2.imread(join(class_img_path, character, "0.png"),0)
		
		rows = 60
		cols = 45
		
		dst = cv2.fastNlMeansDenoising(img, None, 45, 7, 21)
		thresh, im_bw = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		bw_array = im_bw.flatten()
		
		bw_array[bw_array > 250] = 1
		
		modifiedY.append(bw_array)
		
	return inputX, modifiedY
		
def trainCluster() :
	narray = readFromFile('processed_input_cluster.csv')
	#np.random.shuffle(narray)
	X, Y = processInput(narray)
	hidden_layer_array = [2700]
	
	nn = nl.NN()
	nn.train(X, Y, hidden_layer_array, learning_rate=0.1, number_of_output_nodes=len(Y[0]), total_iterations=50000, print_error_iters=10, saveAtInterval=True, forceTrain=True)
	return nn
	
def test() :
	test_folder = 'classified\\testimages'
	testNN = nl.NN().readNNModel('new_model.pkl')
	

	total_count = 0
	for file in listdir(test_folder) :
		full_path = join(test_folder, file)
		img = cv2.imread(full_path,0)
		
		rows = 60
		cols = 180
		
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
		
		str = ""
		for i in range(0,4) :
			#pdb.set_trace()
			cur_img = x[i]
			
			dst = cv2.fastNlMeansDenoising(cur_img, None, 45, 7, 21)
			thresh, im_bw = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

			'''
			kernel1 = np.ones((2,2),np.uint8)
			kernel2 = np.ones((3,3),np.uint8)
			close = cv2.erode(cur_img,kernel1,iterations = 1)
			#dst = cv2.morphologyEx(cur_img, cv2.MORPH_OPEN, kernel)
			close = cv2.dilate(im_bw,kernel2,iterations = 1)
			thresh, close = cv2.threshold(close, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			'''
			
			bw_array = im_bw.flatten()
			
			test_array = np.append(bw_array,[1])
			
			print(test_array)
			
			test_array[test_array > 250] = 1

			output = testNN.testInstance(test_array)
			print(output.round())
			char = "NULL"
			for key, value in numdict.items():
				if np.array_equal(output.round(), value) :
					char = chr(key)
				
			str += char
			total_count += 1
			
			print(char)
			cv2.imshow("some1",im_bw)
			cv2.waitKey()
			
		print("Predicted ", str)
		#print(char)
		cv2.imshow("some",img)
		cv2.waitKey()
			
		
		#input()
	
	print ( "correct count " , accuracy_count , " : total_count " , total_count , " percent " , (accuracy_count/total_count)*100)
	
def testCluster() :
	print("Test cluster")
	train_folder = 'classified\\testimages'
	

	testNN = nl.NN().readNNModel('temp_data.pkl')
	
	captcha_value_list = []

	with open("classified\\testclass.txt","r") as classFile:
		for line in classFile:
			captcha_value_list.append(line.strip())
	
	
	accuracy_count = 0
	total_count = 0
	captcha_correct_count = 0
	for file in listdir(train_folder) :
		#print("=========== Test Image : ", file)
		index = int((file.split('.'))[0])-1000
		full_path = join(train_folder, file)
		img = cv2.imread(full_path,0)
		
		x = cpp.splitImage(img)

		captcha_str = ""
		
		for i in range(0,4) :
			#pdb.set_trace()
			cur_img = x[i]			
			
			bw_array = cur_img.flatten()
			
			test_array = np.append(bw_array,[1])
			
			test_array[test_array > 0] = 1

			output = testNN.testInstance(test_array)
			
			#print(output.round())
			char = "NULL"
			for key, value in numdict.items():
				if np.array_equal(output.round(), value) :
					char = chr(key)
				
			captcha_str += char
			if char == captcha_value_list[index][i] :
			#	print("correct")
				accuracy_count += 1
				
			total_count += 1
			
		print("Predicted  " , captcha_str)
		print("Correct    " , captcha_value_list[index])
		
		if captcha_str == captcha_value_list[index] :
			captcha_correct_count += 1
		#print(char)
		#cv2.imshow("some",img)
		#cv2.waitKey()
			
		
		#input()
	
		print ( "char correct count " , accuracy_count , " : char total_count " , total_count , " captcha correct count ", captcha_correct_count, " percent " , (accuracy_count/total_count)*100)
	
	
def validateModel() :
	train_folder = 'classified\\fullimages'
	testNN = nl.NN().readNNModel('new_model.pkl')
	
	captcha_value_list = []

	with open("classified\\class.txt","r") as classFile:
		for line in classFile:
			captcha_value_list.append(line.strip())
	
	
	accuracy_count = 0
	total_count = 0
	for file in listdir(train_folder) :
		print("=========== Test Image : ", file)
		index = int((file.split('.'))[0])-1
		full_path = join(train_folder, file)
		img = cv2.imread(full_path,0)
		
		rows = 60
		cols = 180
		
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
		
		for i in range(0,4) :
			#pdb.set_trace()
			cur_img = x[i]
			
			dst = cv2.fastNlMeansDenoising(cur_img, None, 45, 7, 21)
			thresh, im_bw = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			bw_array = im_bw.flatten()
			
			test_array = np.append(bw_array,[1])
			
			#print(test_array)
			
			test_array[test_array > 250] = 1

			output = testNN.testInstance(test_array)
			#print(output.round())
			char = "NULL"
			for key, value in numdict.items():
				if np.array_equal(output.round(), value) :
					char = chr(key)
					
			if char == captcha_value_list[index][i] :
			#	print("correct")
				accuracy_count += 1
				
			total_count += 1
			
		#print(char)
		#cv2.imshow("some",img)
		#cv2.waitKey()
			
		
		#input()
	
	print ( "correct count " , accuracy_count , " : total_count " , total_count , " percent " , (accuracy_count/total_count)*100)
	
def validateCluster(testNN) :
	print("Validating")
	train_folder = 'classified\\fullimages'
	
	if testNN is None:
		testNN = nl.NN().readNNModel('temp_data.pkl')
	
	captcha_value_list = []

	with open("classified\\class.txt","r") as classFile:
		for line in classFile:
			captcha_value_list.append(line.strip())
	
	
	accuracy_count = 0
	total_count = 0
	captcha_correct_count = 0
	for file in listdir(train_folder) :
		#print("=========== Test Image : ", file)
		index = int((file.split('.'))[0])-1
		full_path = join(train_folder, file)
		img = cv2.imread(full_path,0)
		
		x = cpp.splitImage(img)

		captcha_str = ""
		
		for i in range(0,4) :
			#pdb.set_trace()
			print("Correct    " , captcha_value_list[index])
			
			cur_img = x[i]			
			
			bw_array = cur_img.flatten()
			
			test_array = np.append(bw_array,[1])
			
			test_array[test_array > 0] = 1

			output = testNN.testInstance(test_array)
			
			output[output > 0] = 255

			img = np.reshape(output, (60,45))
			
			cv2.imshow("some", img)
			cv2.waitKey()
			'''
			#print(output.round())
			char = "NULL"
			for key, value in numdict.items():
				if np.array_equal(output.round(), value) :
					char = chr(key)
				
			captcha_str += char
			if char == captcha_value_list[index][i] :
			#	print("correct")
				accuracy_count += 1
				
			total_count += 1
			
		print("Predicted  " , captcha_str)
		print("Correct    " , captcha_value_list[index])
		
		if captcha_str == captcha_value_list[index] :
			captcha_correct_count += 1
		#print(char)
		#cv2.imshow("some",img)
		#cv2.waitKey()
			
		
		#input()
	
		print ( "char correct count " , accuracy_count , " : char total_count " , total_count , " captcha correct count ", captcha_correct_count, " percent " , (accuracy_count/total_count)*100)
		'''
	
def validateModelAgainstTrainingSet() :
	testNN = nl.NN().readNNModel('model_99.pkl')
	
	captcha_value_list = []

	with open("classified\\class.txt","r") as classFile:
		for line in classFile:
			captcha_value_list.append(line.strip())
	
	accuracy_count = 0
	total_count = 0
	#train_folder = 'C:\\Users\\Administrator\\Desktop\\pyscripts\\captcha\\classified\\testimages'
	train_folder = 'classified\\fullimages'
	for file in listdir(train_folder) :
		print("File ", file)
		index = int((file.split('.'))[0])-1
		
		full_path = join(train_folder, file)
		img = cv2.imread(full_path,0)
		
		rows = 60
		cols = 180
	
			
		if img.shape[0] < rows :
			diff = rows - img.shape[0]
			zeros = np.zeros((diff,img.shape[1]),dtype=np.uint8)	# zeros uint8 is crucial, otherwise you will just see a black image in imshow . 
			img = np.r_[img,zeros]
		elif img.shape[0] > rows : 
			img = img[:rows]
			
		if img.shape[1] < cols :
			diff = cols - img.shape[1]
			zeros = np.zeros((img.shape[0],diff),dtype=np.uint8)
			img = np.c_[img,zeros]
			
		elif img.shape[1] > cols : 
			img = img[:,:cols]
			
		x = np.array_split(img,4,1)
		
		for i in range(0,4) :
			#pdb.set_trace()
			cur_img = x[i]
			
			dst = cv2.fastNlMeansDenoising(cur_img, None, 45, 7, 21)
			thresh, im_bw = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			bw_array = im_bw.flatten()
			
			test_array = np.append(bw_array,[1])
			
			print(test_array)
			
			test_array[test_array > 250] = 1

			output = testNN.testInstance(test_array)
			print(output.round())
			char = "NULL"
			for key, value in numdict.items():
				if np.array_equal(output.round(), value) :
					char = chr(key)
			
			print("predicted value ", char)
			print("correct value ", captcha_value_list[index][i])
			
			if char == captcha_value_list[index][i] :
				accuracy_count += 1
			
			print(char)
			total_count += 1
			cv2.imshow("some",im_bw)
			cv2.waitKey()
			

			print ( "correct count " , accuracy_count , " : total_count " , total_count , " percent " , (accuracy_count/total_count)*100)
		#input()
	
	
def validateModelAgainstDividedTrainingSet() :
	testNN = nl.NN().readNNModel('model_99.pkl')
	

	train_folder = 'classified\\divided'
	#train_folder = 'C:\\Users\\Administrator\\Desktop\\pyscripts\\captcha\\images'
	
	for folder in listdir(train_folder) :
		classfolder = join(train_folder, folder)
		
		for file in listdir(classfolder) :
			full_path = join(classfolder, file)
			img = cv2.imread(full_path,0)
			
			rows = 60
			cols = 180
		
			cur_img = img
				
			dst = cv2.fastNlMeansDenoising(cur_img, None, 45, 7, 21)
			thresh, im_bw = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			bw_array = im_bw.flatten()
			
			test_array = np.append(bw_array,[1])
				
			print(test_array)
			
			test_array[test_array > 250] = 1

			output = testNN.testInstance(test_array)
			
			print(output.round())
			char = "NULL"
			for key, value in numdict.items():
				if np.array_equal(output.round(), value) :
					char = chr(key)
					
			print("Predicted character " , char)
			cv2.imshow("some",im_bw)
			cv2.waitKey()
			
			
			#input()
	
	
if __name__ == "__main__":
	action = sys.argv[1]
	print(action)
	if action == "train" :
		train()
	elif action == "traincluster" :
		testnn = trainCluster()
		validateModelCluster(testnn)
	elif action == "validate":
		#test()
		#validateModelAgainstTrainingSet()
		validateModel()
		#validateModelAgainstDividedTrainingSet()
	elif action == "test" :
		test()
	elif action == "testcluster" :
		testCluster()
	elif action == "validatecluster" :
		validateCluster(None)
	
	