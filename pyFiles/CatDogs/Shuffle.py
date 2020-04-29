import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle


DATADIR = "/home/askio/Desktop/pyFiles/CatDogs/PetImages" # Directory of images
CATEGORIES = ["Dog", "Cat"] # Self explanatory
IMG_SIZE = 50 # Reduces size of pictures


training_data = [] # Creates array

def create_training_data():
	for category in CATEGORIES: # Self explanatory
		path = os.path.join(DATADIR, category) # Gets path
		class_num = CATEGORIES.index(category) # Assign numbers to each value in the category
		for img in os.listdir(path): # Gets images in each category 
			try:
				img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # Converts to gray scale 
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # Resizes the photo
				training_data.append([new_array, class_num]) # Adds it to the training data array
			except Exception as e: # OS errors ignore kek
				pass

create_training_data() 
random.shuffle(training_data)
print(len(training_data))
for sample in training_data:
	print(sample[1])
X = []
y = []

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

