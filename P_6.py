import cv2
import tensorflow as tf

CATEGORIES = ["Dog","Cat"]

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def showImg(filepath):
   img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
   return img_array

model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('outside_dog_cat/outside_dog.jpg')])
print(prediction)
print(prediction[0][0])
print(int(prediction[0][0]))
print(CATEGORIES[int(prediction[0][0])])
showImg_dog = showImg('outside_dog_cat/outside_dog.jpg')
cv2.imshow('dog', showImg_dog)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("-----------------------")

prediction = model.predict([prepare('outside_dog_cat/outside_cat.jpg')])
print(prediction)
print(prediction[0][0])
print(int(prediction[0][0]))
print(CATEGORIES[int(prediction[0][0])])
showImg_cat = showImg('outside_dog_cat/outside_cat.jpg')
cv2.imshow('cat', showImg_cat)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("-----------------------")


