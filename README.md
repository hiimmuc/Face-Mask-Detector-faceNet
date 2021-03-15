<h1 align="center">Face Mask Detection</h1>     
<div align= "center"><img src="https://github.com/hiimmuc/Face_mask_detction_vn/blob/main/Screenshot%202021-03-13%20152218.jpg" width="200" height="200"/>
  <h4>Face Mask Detection system built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts in order to detect face masks in static images as well as in real-time video streams.</h4>
</div>

# Face Mask Detetor😷
# Including executable file for Windows user

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow+v2.4.1-yellow.svg)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/phgnam-dang/)


## :innocent: Motivation
In the present scenario due to Covid-19, there is no efficient face mask detection applications which are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Also, the absence of large datasets of __‘with_mask’__ images has made this task more cumbersome and challenging. 


## What I did??
- Using facial landmark
- then Using fine tuning with Mobilenetv2 architecture to classify 2 types
- Using PyQt5 to create gui and convert to .exe file by pyinstaller
 
## :hourglass: Project Demo
:movie_camera: [YouTube Demo Link](https://www.youtube.com/watch?v=6gkFgv-RpXY)

🎥r: [Youtube](https://www.youtube.com/watch?v=oVlOWfg4hQA)

## :warning: Framework used
- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
## :star: Features
Our face mask detector didn't use any morphed masked images dataset. The model is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient and thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.).

This system can therefore be used in real-time applications which require face-mask detection for safety purposes due to the outbreak of Covid-19. This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

This system can detect and classify three types: masked, unmasked and incorrectly masked

## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here]()

## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/hiimmuc/Face_mask_detction_vn.git
```

2. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```
3. Install .exe file with this link
```
$ https://drive.google.com/drive/folders/1uCUeixNxNFYcKXnO4dkaDV_UmtcW-Kzk?usp=sharing
```
## :bulb: Working

1. Open terminal. After cloning my respo, go with this command to run the detector
```
$ python3 main.py
```

2. or you cand use this command for the simplier version:
```
$ python3 main02.py
```

3. To detect face masks in real-time video streams type the following command:
```
$ python3 facedetect_yolo.py 
```

4. To see the ui design type:
```
$ python3 GUI.py 
```

## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: hiimmuc1811@gmail.com
or my facebbok:
https://www.facebook.com/phgnam1811/


## :raising_hand: Citation

You are allowed to cite any part of the code or our dataset. You can use it in your Research Work or Project. Remember to provide credit to the Maintainer Bernard Deng by mentioning a link to this repository and her GitHub Profile.

Follow this format:
- Author's name - Bernard Deng
- Date of publication or update in parentheses.
- Title or description of document.
- URL.

## 👏 References:
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
https://www.tensorflow.org/tutorials/images/transfer_learning
