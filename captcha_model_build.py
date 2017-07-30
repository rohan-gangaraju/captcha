import numpy as np
import cv2

from os import listdir, makedirs
from os.path import join,isfile,exists

import csv
import neurallib as nl

import sys
#import pdb

import captcha_pre_process as cpp

#np.set_printoptions(threshold=np.inf) #setting this may cause program to hang while trying to print some array

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
		
def getImageClassDict() :
	img_class_dict = {}
	
	train_images_path = 'classified\\image_classes'

	onlyfiles = [f for f in listdir(train_images_path)]

	for file in onlyfiles :
		img_path = join(train_images_path, file, "0.png");
		img = cv2.imread(img_path,0)
		
		dst = cv2.fastNlMeansDenoising(img, None, 45, 7, 21)
		thresh, im_bw = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		bw_array = im_bw.flatten()
		
		bw_array[bw_array > 250] = 1
		bw_array[bw_array < 1] = 0
		
		img_class_dict[file] = bw_array
		
	return img_class_dict
	
def train() :
	narray = readFromFile('processed_input_cluster.csv')
	#np.random.shuffle(narray)
	X, Y = processInput(narray)
	hidden_layer_array = [2700]
	
	nn = nl.NN()
	nn.train(X, Y, hidden_layer_array, learning_rate=0.1, number_of_output_nodes=len(Y[0]), total_iterations=50000, print_error_iters=10, saveAtInterval=True, forceTrain=True)
	return nn

def test() :
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
	
	img_class_dict = getImageClassDict()
		
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
			
			
			output[output < 1] = 0
			
			match_sum = 0
			output_char = "NULL"
			
			for key,value in img_class_dict.items() :
				#print(key)
				match_count = np.sum(output == value)
				if match_count > match_sum :
					output_char = key
					match_sum = match_count
					#print("match" , key)
					#input()

			#output[output == 1] = 255
			#img = np.reshape(output, (60,45))
			#cv2.imshow("some", img)
			#cv2.waitKey()
			
			#print("Correct    " , captcha_value_list[index])
			#print("Predicted  " , output_char)
			captcha_str += output_char
			if output_char == captcha_value_list[index][i] :
			#	print("correct")
				accuracy_count += 1
				
			total_count += 1
			
		print("Predicted  " , captcha_str)
		print("Correct    " , captcha_value_list[index])
		
		if captcha_str == captcha_value_list[index] :
			captcha_correct_count += 1
		
		
		print ( "char correct count " , accuracy_count , " : char total_count " , total_count , " captcha correct count ", captcha_correct_count, " percent " , (accuracy_count/total_count)*100)
			
			
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

def validate(testNN) :
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
	
	img_class_dict = getImageClassDict()
	
	for file in listdir(train_folder) :
		#print("=========== Test Image : ", file)
		index = int((file.split('.'))[0])-1
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
			
			output[output < 1] = 0
			
			match_sum = 0
			output_char = "NULL"
			
			for key,value in img_class_dict.items() :
				#print(key)
				match_count = np.sum(output == value)
				if match_count > match_sum :
					output_char = key
					match_sum = match_count
					#print("match" , key)
					#input()

			#output[output == 1] = 255
			#img = np.reshape(output, (60,45))
			#cv2.imshow("some", img)
			#cv2.waitKey()
			
			#print("Correct    " , captcha_value_list[index])
			#print("Predicted  " , output_char)
			captcha_str += output_char
			if output_char == captcha_value_list[index][i] :
			#	print("correct")
				accuracy_count += 1
				
			total_count += 1
			
		print("Predicted  " , captcha_str)
		print("Correct    " , captcha_value_list[index])
		
		if captcha_str == captcha_value_list[index] :
			captcha_correct_count += 1
		
		
		print ( "char correct count " , accuracy_count , " : char total_count " , total_count , " captcha correct count ", captcha_correct_count, " percent " , (accuracy_count/total_count)*100)


if __name__ == "__main__":
	action = sys.argv[1]
	print(action)
	if action == "train" :
		train()
	elif action == "validate":
		validate(None)
	elif action == "test" :
		test()