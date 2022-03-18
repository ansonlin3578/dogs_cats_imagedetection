import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

#----------example for input data------------
# DATADIR = "Datasets/PetImages"
# CATEGORIES = ["Dog", "Cat"]

# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)#path to cat or dog dir
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap = "gray")
#         plt.show()
#         break
#     break

# print(img_array.shape)

# IMG_SIZE = 50

# new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
# plt.imshow(new_array, cmap = "gray")
# plt.show()
#----------example for input data------------

DATADIR = "Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]

training_data = []
IMG_SIZE = 50

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)#path to cat or dog dir
        class_num = CATEGORIES.index(category)#get the classification (dog:0 or cat:1)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()

print(len(training_data))
# training_data = np.array(training_data, dtype=object)
# print(training_data.shape)
import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

print(x[0].shape)
print(x[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)#change into tensor class
print(np.array(x).shape)
# print(x[].shape)

# for i in range(10):
#     test_img = x[i]
#     cv2.imshow("test", test_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import pickle
pickle_out = open("x.pickle", "w+b")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "w+b")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("x.pickle","r+b")
x = pickle.load(pickle_in)

pickle_in = open("y.pickle","r+b")
y = pickle.load(pickle_in)









