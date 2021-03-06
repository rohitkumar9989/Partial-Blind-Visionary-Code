# Important Libraries for the programme to run


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow_hub as hub

class Help_blind ():
  def image_tensor (self,image):
    """
    Converts an image to an tensor file
    `*Args`
      `image`: The image is the `image` of the file which has to be converted in to tensors
      
    `Returns`: The array of tensors ehich has been converted by the machine
    """
    img=tf.io.read_file(filename=image,
                        name='image_tensor')
    try:
      img=tf.image.decode_image(img)
    except Exception as e:
      img=tf.image.decode_jpeg(img)
    finally:
      self.img=tf.image.resize(images=img,
                          size=(224, 224),
                          name='resizer')
      return self.img/255.
  def splitter (self,img):
    """
    This function crops the edges of the images for the left centre and
    right position sof the image
    
    `*Args`
      `img`: The `img` is the tensor of the image which has been converted into tensors
      
    `Returns`: An array of tensors which has been spliited in the form of left, centre, right of the image
    
    For example:
    
    `
    #Pass the image tensor function insude the splitter function just to convert the image into tensor
    right, centre, left=self.splitter (self.image_tensor(image))
    plt.imshoe(right)
    plt.figure()
    plt.imshow(centre)
    plt.figure()
    plt.imshow(left)
    
    `
    """
    image_flipped_centre=tf.image.flip_left_right(img)
    #Flipping the image
    #Centre cropping of the image
    image_flipped_centre=tf.keras.layers.Cropping1D(cropping=(90, 1))(image_flipped_centre)
    image_flipped_2=tf.image.flip_left_right(image_flipped_centre)
    image_flipped_2=tf.image.resize(tf.keras.layers.Cropping1D(cropping=(50, 1))(image_flipped_2), size=(224, 224))
    new = tf.keras.layers.Cropping1D(cropping=(150, 1))(img)
    
    #Left side cropping of the image
    new=tf.image.resize(new, size=(224, 224))
    image_flipped_left=tf.image.flip_left_right(img)
    
    #Right side cropping of the image
    image_flipped_left=tf.keras.layers.Cropping1D(cropping=(170, 1))(image_flipped_left)
    image_flipped_left=tf.image.flip_left_right(image_flipped_left)
    image_flipped_left=tf.image.resize(image_flipped_left, size=(224, 224))

    #Returns the cropped image
    return image_flipped_left,image_flipped_2, new
  def __init__ (self, image, save_checkpoint=True, saved_checkpoints=False):
    """
    This class function enables a blind person to
    wlak properly in a open yard with some obstacles,

    *The arguments which has to be passed are*
    ->`image`: This is the live data images that has been captured in the
                camera and the machine learns the patterns and give sout the putputs

    This is not a already trained `model` as the `model` training of the `model` is already
    being done in the `__init__` function 
    """
    self.image=image
    data=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,
                                                     shear_range=0.2,
                                                     height_shift_range=0.2,
                                                     width_shift_range=0.2,
                                                     zoom_range=0.2,
                                                     horizontal_flip=False)
    data=data.flow_from_directory(directory='/content/drive/MyDrive/Data_for_blind',
                                  target_size=(224, 224),
                                  class_mode='categorical',
                                  batch_size=10,
                                  shuffle=False)
    #Getting the model
    base_model=hub.KerasLayer(handle='https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5', trainable=False)

    #Training the model
    self.model_true=tf.keras.models.Sequential([
                                       base_model,
                                       tf.keras.layers.Dense(21, activation='relu'),
                                       tf.keras.layers.Dense(14, activation='relu'),
                                       tf.keras.layers.Dense(len(os.listdir('/content/drive/MyDrive/Data_for_blind')), activation='softmax')
    ])
    
    """
    The learning rate of the Adam is taken with the help of the callbacks of the tensorflow laerning rate scheduler 
    the code goes as follows ans the images are in the images forlder,
    `tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4*10**(epochs/200))`
    
    and plottin g the resluts will show us which is the ideal learning rate
    
    `lrs=1e-4*10**(np.arange(0, 20)/200)
    plt.semilogx(lrs, history.history['loss'])`
    """
    
    self.model_true.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.000131),
                   metrics='accuracy')
    
    #Here the model uses the Learning rate scheduler for the setting its optimial learning rate
    if saved_checpoints=False:
    
      if save_checkpoint=True:
        history=self.model_true.fit(data, epochs=20, steps_per_epoch=len(data), callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4*10**(epochs/200)),
                                                                        tf.keras.callbacks.ModelCheckpoint(filepath='/content/checkpoints.ckpt',
                                                                                                           verbose=1,
                                                                                                           save_weight_only=True,
                                                                                                           save_freq='epoch')],
                               batch_size=20)
      else:
        history=self.model_true.fit(data, epochs=20, steps_per_epoch=len(data), callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4*10**(epochs/200))]
      self.predicte_sides(model=self.model_true)
                                    
                                    
    if svaed_checkpoints=True:
      try:
        self.model_true.load_weights('/content/checkpoints.ckpt')
        self.predicte_sides(model=self.model_true)
      except Exception as e:
        print ("The model has not been saved")
                                   
                                  
                                  
  def predicte_sides(self, model):
    model_true=model
    img=self.image_tensor(image=self.image)
    
    """
    Helps in visualising the correct path with the help
    of splitting the image into three halves,
    the finction is `self.splitter (img)`
    which takes `image` as `img parameter in the function`
    """

    right, centre, left=self.splitter(img)

    classes=sorted(os.listdir('/content/drive/MyDrive/Data_for_blind'))

    #This is a sorted list of the data

    list_dir=[right, centre, left]
    for names in list_dir:
      prediction=model_true.predict(tf.expand_dims(names, axis=0))
      preds=classes[prediction.argmax()]
      plt.title(f'{preds}')
      plt.imshow(preds)
      print (f"{preds}")
      
      
      
 #This function then creates and gives the prediction as divided into three segments, and then predicts the images accordingly,
 #From the models real data test results it gives out the corresct predictions and rarely it provides wrong predcition if models 
 #accuracy is not good and loss rate is not good
 
