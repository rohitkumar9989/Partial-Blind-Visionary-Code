import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def image_tensor (image):
  img=tf.io.read_file(filename=image,
                      name='image_tensor')
  try:
    img=tf.image.decode_image(img)
  except Exception as e:
    img=tf.image.decode_jpeg(img)
  finally:
    img=tf.image.resize(images=img,
                        size=(224, 224),
                        name='resizer')
    return img/255.
  
  def image_swapper (tensor):
  try:
    image=tf.image.flip_left_right(tensor)
  except Exception as e:
    image=tf.image.flip_left_right(tf.expand_dims(tensor))
  return image


img=tf.io.read_file('image_of_mine_lol.jpg')
img=tf.image.decode_image(img)
img=tf.image.resize(img, size=(224, 224))
img=img/255.
print (len(img))
plt.imshow(img)


image_flipped_centre=tf.image.flip_left_right(img)
image_flipped_centre=tf.keras.layers.Cropping1D(cropping=(90, 1))(image_flipped_centre)
image_flipped_2=tf.image.flip_left_right(image_flipped_centre)
image_flipped_2=tf.keras.layers.Cropping1D(cropping=(50, 1))(image_flipped_2)
plt.imshow(image_flipped_2)


image_flipped=tf.image.flip_left_right(img)
plt.imshow(image_flipped)
image_flipped_true=tf.image.flip_left_right(image_flipped)
plt.imshow(image_flipped_true)


image=tf.image.crop_to_bounding_box(img, 0,0,224,3)
print (len(image))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_2=image[:,:,0]


plt.imshow(image_2)
print (image_2)
image_right=tf.image.resize(tf.expand_dims(image_2, axis=0), size=(224, 224))
print (image_right)
plt.imshow(image_right)

image_left=tf.image.central_crop(img, 0.6)
plt.imshow(image_left)
new_image=tf.image.resize(image_left, size=(224, 224))
print (new_image)
