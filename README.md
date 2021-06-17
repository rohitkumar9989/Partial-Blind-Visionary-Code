# Partial-Blind-Visionary-Code

>This Programme helps the people who are with parial blindness and although this code is not written in the computer vision but also it works perfectly fine with the real time data


> The files which are important are already listed in the repo

File names | About the folder
------------ | -------------
*Trees* | As the name suggests th folder has images of the trees which could be considered as an obstacle if person comes through the garden
*Tables* | These Images could be usefull when the person is walking in a house
*Stairs* | The stair cases images
*Poles* | This feature is usefull when the person is walking on a footpath
*Pillars* | Also usefull when the person is waling in a basement of a building
*People* | (😂 Yes I have mis spelt the people name in the folder) This is usefull for the persons who don't want to collide witht the people 😁
*Doors* | Here this images are of the doors which are shaped 
*Dogs* | Dogs images
*chairs* | The person can sit on the chairs
*Cars* | Images of the cars approaching towards the camera
*Bikes* | Bike images which are kept on halt
*Free Path* | *Free path* over here says that the person can move forward or any direction suggested by the machine safely


# This programme is usefull for the people who have partial blindness like
	1. Galucoma
	1. Diabeties

# Pre information of the code

> The code doesn't come with any pre trained model as the model training is already present in the class function

> The code Uses Resnet 50 V2 architecture

> The trail and error methods are in the trail folders



# The main things to look after this code are these

> 1. The model takes a long time to be trained hence the model saves the models checkpoints and
> you can process it after the model has been trained for once. the code is provided in the repo for swapping the model training 
> function

> 2. There is no need to make an checkpoints dir. Because the model makes it while the model is being trained for the first time
