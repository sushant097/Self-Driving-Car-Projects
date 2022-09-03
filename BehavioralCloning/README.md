# Behavioral Cloning Project


#### Files
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network modeled after NVIDIA's [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The model includes RELU layers to introduce nonlinearity (first example: model.py line 99), and the data is normalized in the model using a Keras lambda layer (model.py line 98).

#### 2. Attempts to reduce overfitting in the model

The model contains multiple dropout layers between the dense layers in order to reduce overfitting (first example: model.py lines 106). These layers reduce overfitting by randomly dropping outputs from a layer of neural network - resulting in more neurons being forced to learn the characteristics in the network. This helps the model to better generalize on unseen data.

In order to keep track of whether the model is over/under-fitting, the model was trained and validated on different datasets. This helps us to keep track of how well the model performs with previously unseen data (model.py function load_data lines 83).

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually, but was initialized on the default learning rate of 0.0001 (model.py line 169).

#### 4. Appropriate training data

I used the data provided by Udacity as a starting place for training my model. The model architecture and the given data led to the car drifting off the track at the first sharp turn by the lake. Because of this, I decided it was best to include some additional training data focused on recovering from drifting. I captured data of the car recovering from the left and right sides of the road in both the forward and reverse direction on the track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build off existing research completed by experts in the field and fine-tune the model to my data.

My first step was to use a convolution neural network model similar to the NVIDIA [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because it is a similar use case and was suggested in the course notes.

In order to gauge how well the model was training, I split my image and steering angle data into a training and validation set. During training, I found that the model trained only on the supplied data had a low mean squared error (MSE) on the training set but a high MSE on the validation set. This implied the model might be overfitting.

To combat the potential overfitting, I modified the model to include multiple dropout layers with a keep_prob of 0.5.

Then I captured training data in which the car was driven the reserve direction around the course. Additionally, I included a random augmentation function which randomly selected images in the training set to be flipped horizontally or had the image's brightness adjusted randomly.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I captured recovery data from both the left and right sides of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 96-111) consisted of a convolution neural network with the following layers and layer sizes:

```
- Cropping2D (removing unneeded top/bottom of the image)
- Lambda (normalization of the image)
- Conv2D(24, (5, 5), strides=(2, 2), activation='relu')
- Dropout(0.5)
- Conv2D(36, (5, 5), strides=(2, 2), activation='relu')
- Conv2D(48, (5, 5), strides=(2, 2), activation='relu')
- Dropout(0.5)
- Conv2D(64, (3, 3), activation='relu')
- Conv2D(64, (3, 3), activation='relu')
- Dropout(0.5)
- Flatten()
- Dense(100)
- Dense(50)
- Dropout(0.5)
- Dense(10)
- Dense(1)
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the data provided by Udacity:

![good_driving_c](https://user-images.githubusercontent.com/11286381/50926331-9e068780-1409-11e9-8cbb-61b83dd3c493.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct toward the center of the lane. These images show what a recovery looks:

![recovery_l](https://user-images.githubusercontent.com/11286381/50926333-9e9f1e00-1409-11e9-85fb-dccb8602d22e.jpg)
![recovery_r](https://user-images.githubusercontent.com/11286381/50926334-9e9f1e00-1409-11e9-9d3e-f7e6a764e1cb.jpg)

To augment the data sat, I also flipped images and angles thinking that this would help the model generalize better. For example, here is an image that has then been flipped:

![recovery](https://user-images.githubusercontent.com/11286381/50926336-9e9f1e00-1409-11e9-91a2-f8996e35ccdd.jpg)
![recovery_flip](https://user-images.githubusercontent.com/11286381/50926332-9e9f1e00-1409-11e9-8252-40803d38b82c.jpg)

Here is a view of the loss history during training.

![history](https://user-images.githubusercontent.com/11286381/50938054-5b55a700-142b-11e9-9aa8-6a233fb03df6.png)

