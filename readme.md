
# Self-Driving Car  🚘 🛣️

This repository contains many autonomous vehicle-related projects I've worked on. This might serve as the foundation for a real autonomous vehicle.

# Projects

## Basic Lane Line Detection
Employ region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Transform line detection to identify lane lines on the road in an image.

![example](https://user-images.githubusercontent.com/11286381/51013469-73a2f000-1517-11e9-922e-a612674272f1.gif)  
_Simple linearly extrapolated lane detections_


[Source](https://github.com/sushant097/Self-Driving-Car-Projects/blob/master/BasicLaneLineDetection)


## Advanced Lane Line Detection
Find lane markings in images and video using color transformations, gradients, and perspective transformation. Determine the curvature of the lane and the vehicle position with respect to center.

![example](https://user-images.githubusercontent.com/11286381/51013566-093e7f80-1518-11e9-9574-2fdba6eb4f38.gif)  
_Lane detections with curvature and offset_
 
[Source](https://github.com/sushant097/Self-Driving-Car-Projects/blob/master/AdvancedLaneLineDetection)

## Behavioral Cloning
Use Udacity's driving simulator to create a dataset to clone driving behavior by training and validating a model using Keras. The model outputs a steering angle to an autonomous vehicle.

_Autonomus driving in the simulator_  
![example](https://user-images.githubusercontent.com/11286381/51013753-17d96680-1519-11e9-8edf-ea62b5a30771.gif)  

[Source](https://github.com/sushant097/Self-Driving-Car-Projects/blob/master/BehavioralCloning)

## Autopilot
Use End-to-End architecture to train a CNN model that predicts a steering angle, trained on real world dataset.

### Results
![](https://github.com/sushant097/Self-Driving-Car-Projects/blob/master/AutoPilot/file/autopilot.gif)
_Autopilot with steering angle prediction_

[Source](https://github.com/sushant097/Self-Driving-Car-Projects/blob/master/Autopilot/)



### References
* [Udacity Self Driving Car Course](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)
* [Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
* [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)


Feel free to improve this project with pull request. If you face any problem, kindly raise an issue.