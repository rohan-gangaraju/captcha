#Script which extracts the first image out of each class.

from os import listdir, makedirs
from os.path import join,isfile,exists
from shutil import copyfile

def extract_images() :
	train_images_path = '..\\classified\\dividedCluster\\'
	dest_path = '..\\classified\\image_classes'

	onlyfiles = [f for f in listdir(train_images_path)]

	print(onlyfiles)

	for file in onlyfiles :
		src = join(train_images_path,file)
		dest = join(dest_path,file)
		print(dest)
		if not exists(dest):
			makedirs(dest)
		copyfile(join(src,"0.png"), join(dest, "0.png"))
		
def run() :
	extract_images()
	
if __name__ == "__main__" :
	run()