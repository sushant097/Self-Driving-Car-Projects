# Autopilot

This project simulates the autopilot's key function of predicting steering angle using the front picture of the car as input.
More sophisticated autopilots, like as Tesla's autopilot, incorporate object identification, vehicle distance calculation, path planning, traffic-light detection, decision making, and so on. To comprehend their surroundings, advanced autopilots use a range of techniques and sensor inputs, including radar, laser light, GPS, odometry, and computer vision.

### Project Setup
Use `pip install requirements.txt`

### Dataset 🗃️
Download the dataset at [here](https://github.com/SullyChen/driving-datasets) and extract into the repository folder

### Code Details
* `Model.py`: Where End-End-to model is defined. Referenced from [Nvidia End-to-End Model](https://arxiv.org/abs/1604.07316)
* `LoadData.py`: Loads the data from the dataset directory. It makes two pickles file each for images and labels.
* `Train.py`: Train the model
* `inference_image.py`: Run the trained model through the dataset and shows the prediction with steering movement.
* `inferene_video.py`: Run inference on the frames of the video. Can be used with real-time camera feed and get the prediction.

### Project Run
1. First Download the dataset and extract on the  project directory.
2. (Optional) For Training - Run `LoadData.py` which create images and labels in pickle format.
3. (Optional) For training run `train.py`. You can use this [Kaggle Notebook](https://www.kaggle.com/code/sushant097/autopilot) for training. 
4. Run `inference_image.py` for autopilot steering angle prediction.


### Results
![](file/autopilot.gif)

### References
1. Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)

2. This implementation also took a lot of inspiration from the Sully Chen github repository: https://github.com/SullyChen/Autopilot-TensorFlow  


Feel free to improve this project with pull request. If you face any problem, kindly raise an issue.
