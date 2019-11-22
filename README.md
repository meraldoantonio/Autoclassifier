# Autoclassifier

## Background
I made this project as a submission to Grab's <a href = "https://www.aiforsea.com/challenges">AI for S.E.A.</a> challenge. The project consists of the automatic classification of unlabeled car images into one of the 196 preset car classes.

There are two ways to run this project: by running it as a locally-served website (recommended) or by running it the provided Jupyter notebook. The instructions for both are provided below.

### Dataset
The main dataset used in this project is the training set of <a href="https://ai.stanford.edu/~jkrause/cars/car_dataset.html">Stanford's Car Dataset.</a> It contains 8,144 images of 196 classes of cars. It also contains the coordinates of the car's bounding box, which indicates the location of the car in the image.

### Preprocessing
Prior to training, several preprocessing steps were performed.  Firstly. the dataset was split into training, validation and testing sets with a ratio of 6:3:1. Next, the non-car-containing parts of the images were removed using the bounding box coordinates provided. This cropping step was done so that the model only learns to detect and classify cars. The cropped image was then resized to 224 x 224 pixels.

### Initial Classification

Transfer learning was used to create the first model, a <a href="https://arxiv.org/abs/1512.03385">ResNet-152 model.</a>  ResNet-152 model contains residual connections that enable it to train well despite its substantial depth. The code and pre-trained model weights for the Keras implementation of ResNet-152 were downloaded from the <a href = "https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6">GitHub page</a> of Felix Yu. After 136 epochs, a validation accuracy of around 88% was obtained.

### Object Detection Model

One of the reasons that enabled the initial model to achieve a decent performance was the fact that it was trained and tested using cropped images that only contain cars. However, unseen car images to which the model will be applied do not come with bounding box coordinates and might contain other objects that would confuse the model

To improve the overall prediction, an object detection model will be deployed in conjunction with the classification model. This model allows the pinpointing of a car's location in an image and the subsequent cropping of its non-car-containing parts. In this project, a pre-trained model from <a href="https://github.com/tensorflow/models">Tensorflow's Object Detection API</a> was used. The original Tensorflow model returns the locations of <u>all</u> cars that appear in an image. I adjusted the model so that it will only detect and localize <u>one</u> car - the most prominently displayed one.

### Data Augmentation and Model Retraining

Another strategy I employed to increase the model's accuracy is to augment the dataset. I used Python's Google Image Downloader to automatically download an additional 100 images of each car class.

This image downloading step is not perfect. Some of the searches yielded non-car images, while some car images downloaded are of a "wrong" class, one that is different from the search keywords used. I used the previously described models to weed out these confounding images. First, the object detection model was employed to ensure that downloaded images indeed contain a car; images that don't contain any car were discarded. Next, I used the classification model to filter out falsely labeled images;  I only took in images whose predictions match the search keywords I used to get the images. After this filtering step, around 70 new images were obtained for each class

These new images from Google were combined together with the training set from the original Stanford Dataset. The same validation and testing set were used. The classification model was then retrained from scratch. After 392 epochs, an accuracy of around 94% was obtained.


## Website

I served the models in a website created using Flask and Dash.

![image](https://drive.google.com/uc?export=view&id=1MmAgeqCKAjIR2wxG4TavyMgXerfLElvS)

To run the website locally, please perform the following steps. Note: If possible, please run this step in Linux systems (Ubuntu/Pop OS); I have yet to test this in other operating systems.

1. Clone this repository into a folder in your machine
```
git clone https://github.com/meraldoantonio/Autoclassifier.git
cd Autoclassifier
```
2. Create a virtual environment and install the dependencies within the virtual environment
```
conda create -n autoclassifierenv python=3.6
conda activate autoclassifierenv
pip install -r requirements.txt
```

3. Run Flask by running the following command inside the repository
```
flask run
```
This step might take a while (up to 120 seconds) as the models have to be downloaded and loaded onto the backend server.

4. Once the server is done loading, you can open localhost, typically http://localhost:5000. Open the website in a browser (preferably Google Chrome) and view the website on a fullscreen mode.

