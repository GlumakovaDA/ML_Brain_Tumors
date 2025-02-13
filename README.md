<h1>Brain Tumors clasifier using VGG16</h1><br>  
This project implements a convolutional neural network (CNN) based on VGG16 to classify brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary Tumor. The model is trained on 
MRI images and utilizes transfer learning for improved accuracy. <br>
<h2>Dataset</h2><br>
The dataset consists of MRI images of brain tumors, categorized into: <br>
 <br>
Glioma <br>
Meningioma <br>
No Tumor <br>
Pituitary Tumor <br>
 <br>
<h2>Data Structure</h2> <br>
Train/ - Training images <br>
Val/ - Validation images <br>
Each class is stored in a separate folder. <br>
<br>
<h2>Model Architecture</h2><br>
Base Model: Pretrained VGG16 with imagenet weights.<br>
Modifications:<br>
GlobalAveragePooling2D<br>
Dense(1024, activation='relu')<br>
Dense(4, activation='softmax') (output layer for 4 classes)<br>
The model is trained for 10 epochs<br>
<br>
<h2>Results</h2><br>


![image](https://github.com/user-attachments/assets/b7762534-4057-49b4-b80c-b9e318f28b63)<br>

![image](https://github.com/user-attachments/assets/2c294f49-dfa9-41dd-8514-a49e95cf889f)<br>


```
============TEST METRICS=============
Accuracy: 85.15625%
          precision    recall    f1-score   support  

Glioma       0.86      0.79      0.83       136  
Meningioma   0.77      0.83      0.80       140
No Tumor     0.93      0.80      0.86       100
Pituitary    0.87      0.97      0.92       136
accuracy                         0.85       512  
macro avg    0.86      0.85      0.85       512
weighted avg 0.85      0.85      0.85       512
```
