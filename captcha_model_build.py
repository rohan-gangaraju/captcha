import numpy as np
import cv2

from os import listdir, makedirs
from os.path import join,isfile,exists

import csv
import neurallib as nl

import sys

import captcha_pre_process as cpp

#np.set_printoptions(threshold=np.inf) #setting this may cause program to hang while trying to print some array

def readFromFile(file) :
	return np.genfromtxt(file, delimiter=",")
	
def processInput(narray) :
	'''
		narray - [50, 0, 0, 0, 1, 1, .....]
		
		Need to split the first column, convert the ascii value '50' to a character '2' and get the image array for the character class.
		
	'''
	
	
	inputY, inputX = np.split(narray,[1], axis=1)

	modifiedY = []

	for i in range(0,len(inputY)) : 
		character = chr(inputY[i]);
		#print("Instance character : ", character)
		
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
	print("Starting training")
	
	# Read already processed input data
	narray = readFromFile('processed_input_cluster.csv')
	
	# X indicates the input node activation for each input image and Y indicates the required output node activation indicating the image array for the required class
	X, Y = processInput(narray)
	
	# Set the NN hidden layer structure
	hidden_layer_array = [2700]
	
	# Train the NN model with the required parameters
	nn = nl.NN()
	nn.train(X, Y, hidden_layer_array, learning_rate=0.1, number_of_output_nodes=len(Y[0]), total_iterations=50000, print_error_iters=10, saveAtInterval=True, forceTrain=True)
	
	test()
	return nn

def test() :
	print("Testing model")
	
	train_folder = 'classified\\testimages'
	
	# Get the mapping between class character and image array { '2' : [1,0,...], '3' : [0,0,..], ... }
	img_class_dict = getImageClassDict()
		
	# Read trained model
	testNN = nl.NN().readNNModel('temp_data.pkl')
	
	captcha_value_list = []

	with open("classified\\testclass.txt","r") as classFile:
		for line in classFile:
			captcha_value_list.append(line.strip())
	
	char_correct_count = 0
	char_total_count = 0
	captcha_correct_count = 0
	captcha_total_count = 0
		
	for file in listdir(train_folder) :
		#print("=========== Test Image : ", file)
		index = int((file.split('.'))[0])-1000
		full_path = join(train_folder, file)
		img = cv2.imread(full_path,0)
		
		x = cpp.splitImage(img)

		captcha_str = ""
		
		# Divide the 4 character captcha into each character and use the NN model to predict each character
		for i in range(0,4) :
			cur_img = x[i]			
			
			bw_array = cur_img.flatten()
			
			test_array = np.append(bw_array,[1])
			
			test_array[test_array > 0] = 1
			
			# Predict the output character using the NN model
			output = testNN.testInstance(test_array)
			
			output[output < 1] = 0
			
			match_sum = 0
			output_char = "NULL"

			# Simple array comparison between predicted output activation array and the image class array which is already collected the image dict
			for key,value in img_class_dict.items() :
				match_count = np.sum(output == value)
				if match_count > match_sum :
					output_char = key
					match_sum = match_count

			'''
				# Uncomment to see the character image that is generated using the NN model
				
				output[output == 1] = 255
				img = np.reshape(output, (60,45))
				cv2.imshow("some", img)
				cv2.waitKey()
			'''
			
			captcha_str += output_char
			if output_char == captcha_value_list[index][i] :
			#	print("correct")
				char_correct_count += 1
				
			char_total_count += 1
			
		print("Predicted  " , captcha_str)
		print("Correct    " , captcha_value_list[index])

		captcha_total_count +=1
		
		if captcha_str == captcha_value_list[index] :
			captcha_correct_count += 1
		
		summary_string = """
		
		Single Character 
			Correct count : %s
			Total count   : %s

			Percentage    : %s
			
		Captcha
			Correct count : %s
			Total count   : %s		
		
			Percentage    : %s
		""" % (char_correct_count, char_total_count, (char_correct_count/char_total_count)*100, captcha_correct_count, captcha_total_count, (captcha_correct_count/captcha_total_count)*100)
		
		print (summary_string)

def validate(testNN) :
	print("Validating using the training set")
	
	# Get the mapping between class character and image array { '2' : [1,0,...], '3' : [0,0,..], ... }
	img_class_dict = getImageClassDict()
		
	train_folder = 'classified\\fullimages'
	
	# Read trained model
	if testNN is None:
		testNN = nl.NN().readNNModel('temp_data.pkl')
	
	captcha_value_list = []

	with open("classified\\class.txt","r") as classFile:
		for line in classFile:
			captcha_value_list.append(line.strip())
	
	
	char_correct_count = 0
	char_total_count = 0
	captcha_correct_count = 0
	captcha_total_count = 0	

	for file in listdir(train_folder) :
		#print("=========== Test Image : ", file)
		index = int((file.split('.'))[0])-1
		full_path = join(train_folder, file)
		img = cv2.imread(full_path,0)
		
		x = cpp.splitImage(img)

		captcha_str = ""
		
		# Divide the 4 character captcha into each character and use the NN model to predict each character
		for i in range(0,4) :
			cur_img = x[i]			
			
			bw_array = cur_img.flatten()
			
			test_array = np.append(bw_array,[1])
			
			test_array[test_array > 0] = 1
			
			# Predict the output character using the NN model
			output = testNN.testInstance(test_array)
			
			output[output < 1] = 0
			
			match_sum = 0
			output_char = "NULL"

			# Simple array comparison between predicted output activation array and the image class array which is already collected the image dict
			for key,value in img_class_dict.items() :
				match_count = np.sum(output == value)
				if match_count > match_sum :
					output_char = key
					match_sum = match_count

			'''
				# Uncomment to see the character image that is generated using the NN model
				
				output[output == 1] = 255
				img = np.reshape(output, (60,45))
				cv2.imshow("some", img)
				cv2.waitKey()
			'''
			
			captcha_str += output_char
			if output_char == captcha_value_list[index][i] :
			#	print("correct")
				char_correct_count += 1
				
			char_total_count += 1
			
		print("Predicted  " , captcha_str)
		print("Correct    " , captcha_value_list[index])

		captcha_total_count +=1
		
		if captcha_str == captcha_value_list[index] :
			captcha_correct_count += 1
		
		summary_string = """
		
		Single Character 
			Correct count : %s
			Total count   : %s

			Percentage    : %s
			
		Captcha
			Correct count : %s
			Total count   : %s		
		
			Percentage    : %s
		""" % (char_correct_count, char_total_count, (char_correct_count/char_total_count)*100, captcha_correct_count, captcha_total_count, (captcha_correct_count/captcha_total_count)*100)
		
		print (summary_string)
		
if __name__ == "__main__":
	action = sys.argv[1]
	#print(action)
	if action == "train" :
		train()
	elif action == "validate":
		validate(None)
	elif action == "test" :
		test()