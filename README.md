# Image Classification with MobileNetV2

This project uses the pre-trained MobileNetV2 model to classify images into categories based on the ImageNet dataset. The model makes predictions for various objects or scenes in the image, leveraging TensorFlow and Keras.


[Overview](#overview)

[Techniques Covered](#techniques-covered)

[Features](#features)

[Usage](#usage)

[Dependencies](#dependencies)

[Results](#results)


## OverView
The goal of this project is to classify images using the pre-trained MobileNetV2 model. MobileNetV2 is a lightweight deep learning model, efficient for mobile and embedded vision applications. The model is trained on the ImageNet dataset, which includes a wide range of objects and scenes. The project fetches an image from a URL, preprocesses it, and makes a classification prediction, displaying the top predicted categories.

## Techniques Covered
- MobileNetV2: Pre-trained deep learning model for image classification.
- Image Preprocessing: Resizing and normalizing images to fit the model’s input requirements.
- Prediction: Using the model to predict the image's class based on ImageNet's 1,000 classes.
- TensorFlow/Keras: Frameworks used to load the model and perform inference.

## Features
- Image URL Input: Allows users to input an image URL for classification.
- Pre-trained Model: Utilizes MobileNetV2 pre-trained on ImageNet for object classification.
- Top N Predictions: Displays the top 3 predicted labels with their confidence scores.
- Matplotlib Visualization: Displays the input image with predicted labels.

## Usage
- Fetch Image from URL: Input an image URL, and the image will be fetched from the web.
- Preprocess Image: Resize and normalize the image to fit the MobileNetV2 model input.
- Classify the Image: Use MobileNetV2 to classify the image and return the top 3 predicted labels.
- Display Results: Show the image with the predicted labels and confidence scores.

## Step-by-step
- Enter the Image URL: Provide the image URL for the model to classify.
- Preprocess the Image: The image is resized to 224x224 pixels and normalized.
- Model Classification: The MobileNetV2 model is used to predict the top 3 classes for the image.
- Display Results: The image and the top 3 predicted classes are displayed.

## Dependencies
```
requests         # Fetching image from the URL
tensorflow       # TensorFlow for loading and using the MobileNetV2 model
matplotlib       # Displaying the image and results
numpy            # Numerical computations
Pillow           # Image processing

```
## Results
- Prediction Accuracy: The accuracy is based on the top-1 predicted label from MobileNetV2.
- Top 3 Predictions: The model outputs the top 3 most likely labels along with confidence scores.

### Sample Output

#### Image Display
The uploaded image is displayed with a predicted label and confidence score.

#### Top 3 Predictions
```
Top 3 Predictions:
1. French Bulldog: 60.53%
2. Chihuahua: 28.32%
3. Pug: 8.14%
```

