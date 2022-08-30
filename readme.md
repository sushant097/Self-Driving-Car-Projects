# Udacity Self-Driving Car Engineer Nanodegree

This repository houses my solutions for projects completed as part of Udacity's [Self-driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).


## Projects

### Basic Lane Line Detection
Employ region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Transform line detection to identify lane lines on the road in an image.

![example](https://user-images.githubusercontent.com/11286381/51013469-73a2f000-1517-11e9-922e-a612674272f1.gif)  
_Simple linearly extrapolated lane detections_


[Rendered notebook](http://nbviewer.jupyter.org/github/markmisener/udacity-self-driving-car-engineer/blob/master/p1-find-lane-lines/P1.ipynb)  
[Project writeup](https://github.com/markmisener/udacity-self-driving-car-engineer/blob/master/p1-find-lane-lines/writeup.md)  
[Source](https://github.com/markmisener/udacity-self-driving-car-engineer/tree/master/p1-find-lane-lines)


### Advanced Lane Line Detection
Find lane markings in images and video using color transformations, gradients, and perspective transformation. Determine the curvature of the lane and the vehicle position with respect to center.

![example](https://user-images.githubusercontent.com/11286381/51013566-093e7f80-1518-11e9-9574-2fdba6eb4f38.gif)  
_Lane detections with curvature and offset_


[Rendered notebook](http://nbviewer.jupyter.org/github/markmisener/udacity-self-driving-car-engineer/blob/master/p2-advanced-lane-line-detection/P2.ipynb)  
[Project writeup](https://github.com/markmisener/udacity-self-driving-car-engineer/blob/master/p2-advanced-lane-line-detection/writeup.md)  
[Source](https://github.com/markmisener/udacity-self-driving-car-engineer/blob/master/p2-advanced-lane-line-detection/)

### Traffic sign classifier
Train and validate a deep learning model using TensorFlow to classify traffic sign images using the German Traffic Sign Dataset.

[Rendered notebook](https://nbviewer.jupyter.org/github/markmisener/udacity-self-driving-car-engineer/blob/master/p3-traffic-sign-classifier/Traffic_Sign_Classifier.ipynb)  
[Project writeup](https://github.com/markmisener/udacity-self-driving-car-engineer/blob/master/p3-traffic-sign-classifier/writeup.md)  
[Source](https://github.com/markmisener/udacity-self-driving-car-engineer/tree/master/p3-traffic-sign-classifier)  

### Behavioral Cloning
Use Udacity's driving simulator to create a dataset to clone driving behavior by training and validating a model using Keras. The model outputs a steering angle to an autonomous vehicle.

![example](https://user-images.githubusercontent.com/11286381/51013753-17d96680-1519-11e9-8edf-ea62b5a30771.gif)  
_Autonomus driving in the simulator_  

[Project writeup](https://github.com/markmisener/udacity-self-driving-car-engineer/blob/master/p4-behavioral-cloning/writeup.md)  
[Source](https://github.com/markmisener/udacity-self-driving-car-engineer/tree/master/p4-behavioral-cloning)

### Extended Kalman Filter
Utilize a Kalman filter, and simulated lidar and radar measurements to track the a bicycle's position and velocity.  

Lidar measurements are red circles, radar measurements are blue circles with an arrow pointing in the direction of the observed angle, and estimation markers are green triangles.

<img width="794" alt="dataset_1" src="https://user-images.githubusercontent.com/11286381/51014070-b1554800-151a-11e9-8690-93b7226af20a.png">  


[Source](https://github.com/markmisener/udacity-self-driving-car-engineer/tree/master/p5-extended-kalman-filters)

### Localization: Particle Filter

A 2 dimensional particle filter in C++. The particle filter is given a map and some initial localization information (analogous to what a GPS would provide). At each time step the filter is also given observation and control data.

![particle_filter](https://user-images.githubusercontent.com/11286381/54099737-ff9a9200-4377-11e9-8027-31408ed82d46.gif)

[Source](https://github.com/markmisener/udacity-self-driving-car-engineer/tree/master/p6-sparse-particle-filters)

### Path Planning: Vehicle Trajectories

Safely navigate a self-driving car around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

![path_planning](https://user-images.githubusercontent.com/11286381/56447254-0e9c2a80-62bc-11e9-8175-d109df365521.gif)

[Video](https://youtu.be/pV1saRkMq9o)  
[Source](https://github.com/markmisener/udacity-self-driving-car-engineer/tree/master/p7-path-planning)

### PID controller

Use a Proportional-Integral-Derivative Controller (PID), to control the steering angle of a simuluated self-driving car maneuvering around a circular track.

![ezgif com-optimize](https://user-images.githubusercontent.com/11286381/56873418-160fb200-69e7-11e9-8bc2-1a71ebebbee3.gif)

[Video](https://youtu.be/TC73FRDNECA)  
[Source](https://github.com/markmisener/udacity-self-driving-car-engineer/tree/master/p8-pid-controller)

### Capstone: Programming a real self-driving car

Write ROS nodes to implement core functionality of the autonomous vehicle system, including traffic light detection, control, and waypoint following. The following is a system architecture diagram showing the ROS nodes and topics used in the project.


![final-project-ros-graph-v2](https://user-images.githubusercontent.com/11286381/59644293-3d4e4a00-9121-11e9-9075-5c076213bb4c.png)
