import numpy as np
#from memory_profiler import profile
#import _pickle as pickle 	
import dill
import time

#np.set_printoptions(threshold=np.inf) #setting this may cause program to hang while trying to print some array

class layer :
	def __init__(self, number_of_instances, number_of_nodes):
		self.nodes = number_of_nodes
		self.narray = np.zeros((number_of_instances,number_of_nodes))
		self.error = np.zeros((number_of_instances,number_of_nodes))
		self.delta = np.zeros((number_of_instances,number_of_nodes))

class synapse :
	def __init__(self, nodes_current_layer, nodes_next_layer):
		self.narray = np.random.random((nodes_current_layer, nodes_next_layer))
		
class NN :

	def __init__(self) :
		self.layers = []
		self.synapses = []
		self.learning_rate = 0.001
		self.number_of_instances = 0
		self.number_of_input_nodes = 0
		self.number_of_hidden_layers = 0
		self.number_of_hidden_nodes_array = []
		self.number_of_output_nodes = 0
		self.total_iterations = 0
		
		self.print_error_iters = 0
		self.currentIteration = 0

	def activate(self, x) :
		return 1/(1+np.exp(-x))
		
	def derive(self, x) :
		return self.activate(x) * (1-self.activate(x))

	def init_neurons(self, number_of_instances, number_of_input_nodes, hidden_layer_array, learning_rate=0.01, number_of_output_nodes=1, total_iterations=50000, print_error_iters=1000) : 
		
		self.number_of_instances = number_of_instances
		self.number_of_input_nodes = number_of_input_nodes
		self.number_of_output_nodes = number_of_output_nodes
		
		self.learning_rate = learning_rate
		self.number_of_hidden_layers = len(hidden_layer_array)
		
		for i in range(0,self.number_of_hidden_layers) :
			self.number_of_hidden_nodes_array.append(hidden_layer_array[i])
			
		self.total_iterations = total_iterations
		self.print_error_iters = print_error_iters
		
		print("Learning Rate : ", self.learning_rate)
		print("Number of hidden layers :  ", self.number_of_hidden_layers)
		print("Total iterations ", self.total_iterations)
		print("Print error interval ", self.print_error_iters)
		
		print("Number of instances ( m ) ", self.number_of_instances)
		print("Number of input nodes ( n ) ", self.number_of_input_nodes)
		
		for i in range(0,self.number_of_hidden_layers) :
			print("Hidden Layer ", i+1, " : Number of hidden nodes ( h ) ", self.number_of_hidden_nodes_array[i])
		
		print("Number of output nodes ( r ) ", self.number_of_output_nodes)

		m = self.number_of_instances
		n = self.number_of_input_nodes
		r = self.number_of_output_nodes
				
		self.layers.append(layer(m,n))	# First layer ( L0 ) with ( n ) neurons and ( m ) instances
		
		for i in range(0,self.number_of_hidden_layers) :
			self.layers.append(layer(m,self.number_of_hidden_nodes_array[i]))	# Hidden layer ( Li ) with ( h ) neurons and ( m ) instances
			
		self.layers.append(layer(m,r))	# Output layer ( L2 ) with ( r ) neurons and ( m ) instances

		for i in range(0, len(self.layers)) :
			print(" Initial Layer [ ", i , " ] \n" , self.layers[i].narray)
				
		for i in range(0, len(self.layers)-1) : 
			self.synapses.append(synapse(self.layers[i].nodes,self.layers[i+1].nodes))
			
		for i in range(0, len(self.synapses)) :
			print(" Initial Synapse [ ", i , " ] \n" , self.synapses[i].narray)
		
	
	def forward_propagate(self) :
		for i in range(1, len(self.layers)):
			self.layers[i].narray = self.activate(self.layers[i-1].narray.dot(self.synapses[i-1].narray))
			#print(" Layer [ ", i , " ] \n" , self.layers[i].narray)
			
	#@profile
	def backward_propagate(self, outputY) :
		number_of_layers = len(self.layers)

		self.layers[-1].error = outputY - self.layers[-1].narray
		
		self.layers[-1].delta = self.learning_rate * (self.layers[-1].error * self.derive(self.layers[-1].narray))
		
		for i in range(number_of_layers-2, 0, -1) : # suppose layers are L0,L1,L2 excluding the final layer, we have to calcuate the error and delta for L1 (number_of_layers-2)=3-2=1
			self.layers[i].error = self.layers[i+1].delta.dot(self.synapses[i].narray.T)
			self.layers[i].delta = self.learning_rate * (self.layers[i].error * self.derive(self.layers[i].narray))
			#print(" Layer [ ", i , " ] \n" , self.layers[i].delta)
		
		for i in range(number_of_layers-2, -1, -1) : # suppose layers are L0,L1,L2, we have to collect the synapses for syn1 and syn0 (number_of_layers-2)=3-2=1
			self.synapses[i].narray += self.layers[i].narray.T.dot(self.layers[i+1].delta)
			#print(" Synapse [ ", i , " ] \n" , self.synapses[i].narray)
		
		#input()

	def unison_shuffled_copies(self, a, b):
		from sklearn.utils import shuffle
		s_a, s_b = shuffle(a, b, random_state=0)
		return s_a,s_b

	def train(self, inputX, inputY, hidden_layer_array, learning_rate=0.01, number_of_output_nodes=1, total_iterations=50000, print_error_iters=1000, saveAtInterval=False, forceTrain=False) :

		#existing_self = self.readNNModel()
		existing_self = None
		
		X = np.c_[inputX, np.ones(len(inputX))]	# Modify the input array and add an additional column of 1's for bias
		Y = inputY
		
		number_of_instances = len(X)
		number_of_input_nodes = len(X[1,:])
		
		currentIteration = 0
		if forceTrain == True or existing_self is None or existing_self.currentIteration == 0 :
			self.init_neurons(number_of_instances, number_of_input_nodes, hidden_layer_array, learning_rate, number_of_output_nodes, total_iterations, print_error_iters)
			self.layers[0].narray = X	# Initially set the first layer ( L0 ) to the input matrix ( X )
		else :
			self = existing_self
			currentIteration = self.currentIteration
		
		min_error = 100
		start_time = time.perf_counter()
		constant_iterations = 0
		for j in range(currentIteration, total_iterations) :
			# Shuffling training data after each epoch
			self.unison_shuffled_copies(self.layers[0].narray, Y)
			
			self.forward_propagate()
			self.backward_propagate(Y)
			if(j % print_error_iters) == 0:   # Only #print the error every 10000 steps, to save time and limit the amount of output. 
				error_percent = np.mean(np.abs(self.layers[-1].error)) * 100
				'''
				if error_percent < 10 :
					self.learning_rate = 0.01
					print("Changing learning_rate to : ", self.learning_rate)
				elif constant_iterations > 1000 :
					self.learning_rate = (self.learning_rate/2)
					constant_iterations = 0
					print("Changing learning_rate to : ", self.learning_rate)
				'''
				elapsed_time = time.perf_counter() - start_time
				start_time = time.perf_counter()
				print("Error === %0.2f " % error_percent, " === iterations ", str(j), " === Percentage training completed %0.2f" % ((j/total_iterations)*100), " === Elapsed time %0.2f" % elapsed_time)
				if saveAtInterval and (error_percent < min_error):
					self.currentIteration = j
					self.saveCurrentObj()
					min_error = error_percent
					constant_iterations = 0
				else :
					constant_iterations += print_error_iters
					
					
				if error_percent < 0.1 :
					break
			
		#self.saveCurrentObj()
		return self

	def testInstance(self, test_input) :
		#print("Test input " , test_input)
		layer_activation = np.array(test_input)
		#print("length" , len(self.synapses))
		for i in range(0, len(self.synapses)):
		#	print(self.synapses[i-1].narray)
			layer_activation = self.activate(layer_activation.dot(self.synapses[i].narray))
		
		#print("Test output ", layer_activation)
		
		return layer_activation
		
	def validateModel(self, test_input, test_output) :
		accuracy_count = 0
		for i in range(0,len(test_input)) : 
			#print("Test input instance" , test_input[i])
			activation = self.testInstance(test_input[i])
			
		
			#print("Test output ", activation.round())
			#print("Required output ", test_output[i])
			correct = np.array_equal(activation.round(),test_output[i])
			print(" i : ", i, " Correct ? ", correct)
			#input()
			if correct :
				accuracy_count += 1
		
		accuracy = (accuracy_count/len(test_input)) * 100
		print("Accuracy = ", accuracy)
		
	def getFinalLayerOutput(self) :
		return self.layers[-1].narray

	def saveCurrentObj(self) :
		temp_file = 'temp_data.pkl'
		print("Saving to file ", temp_file)
		with open(temp_file, 'wb') as output:
			#pickle.dump(self, output, 2)
			dill.dump(self, output)

	def readNNModel(self, fileName='temp_data.pkl') :
		print("Reading from file ", fileName)
		try:
			with open(fileName, 'rb') as input:
				#self = pickle.load(input)
				self = dill.load(input)
		except (FileNotFoundError, EOFError) as error :
			print("Wrong file or file path")
		return self
		
def exampleRun() :

	print(" =============== Example Run (XOR) =================")
	X = np.array([[0,0],
			[0,1],
			[1,0],
			[1,1]])
			
	Y = np.array([[0],
			 [1],
			 [1],
			 [0]])

	number_of_output_nodes = 1
	hidden_layer_array = [4,2]
	print(len(hidden_layer_array))
	
	learning_rate = 0.9
	total_iterations = 50000
	print_error_iters = 1000
	
	trainNN = NN()
	
	trainNN = trainNN.train(X, Y, hidden_layer_array, learning_rate, number_of_output_nodes, total_iterations, print_error_iters, saveAtInterval=True, forceTrain=False)
			
	print("Final value of Output Layer " , trainNN.getFinalLayerOutput())
	
	test_input = [[1,0,1]]
	testNN = NN().readNNModel()
	testNN.testInstance(test_input)
	
if __name__ == "__main__" :
	exampleRun()
	