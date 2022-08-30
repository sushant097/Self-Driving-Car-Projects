import cv2
import glob
import random
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 160, 320, 3


def augment(image, measurement):
    """ Randomly augments an image and it's corresponding steering angle.

    Args:
        image (np.array): array representation of an image
        measurement (float): steering angle

    Returns:
        image (np.array): array representation of an image
        measurement (float): steering angle
    """
    # randomly flip the image and measurement
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        measurement = -measurement
    # randomly adjust the brighness of the image
    if np.random.rand() < 0.5:
        delta_pct = random.uniform(0.4, 1.2)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * delta_pct
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image, measurement


def load_image(image_file):
    """ Loads RGB images from a file

    Args:
        image_file (str): name of image file

    Returns:
        numpy array representation of the image
    """
    image_file = image_file.strip()
    if image_file.split('/')[0] == 'IMG':
        image_file = '/opt/carnd_p3/data/{}'.format(image_file)
    return mpimg.imread(image_file)


def load_data(test_size, additional_training_data=False):
    """ Loads the data from a directory and split it into
    training and testing datasets.

    Args:
        test_size (float): percentage of data to hold out for testing
    Returns:
        List containing train-test split of inputs. (X_train, X_val, y_train, y_val)
    """
    header = None
    names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv('/opt/carnd_p3/data/driving_log.csv',
                          header=header, names=names)

    if additional_training_data:
        for filename in glob.glob('training_data/*.csv'):
            tmp_df = pd.read_csv(filename, header=header, names=names)
            data_df = pd.concat([data_df, tmp_df])

    # ignore header rows of additional training data
    data_df = data_df[(data_df.center != 'center') |
                      (data_df.left != 'left') |
                      (data_df.right != 'right')]

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    return train_test_split(X, y, test_size=test_size)


def build_model():
    """ Creates a keras sequential model, derived from NVIDIA's architecture
    (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf),
    with some minor modifications.

    Args:
        None
    Returns:
        keras sequential model
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def batch_generator(image_paths, steering_angles, batch_size):
    """ Creates a generator pull pieces of the data and process them on the fly.

    Args:
        image_paths (np.array): array of paths to image files
        steering_angle (np.array): array of steering angles
        batch_size (int): number of items to process

    Returns:
        images (np.array): array of images
        measurements (np.array): array of steering angles
    """
    images = np.empty([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    measurements = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            image = load_image(center)
            measurement = float(steering_angles[index])

            image, measurement = augment(image, measurement)

            images[i] = image
            measurements[i] = measurement

            i += 1
            if i == batch_size:
                break
        yield images, measurements


def main():
    """Entry point for training the model"""

    # Hyperparameters
    test_size = 0.20
    batch_size = 256
    epochs = 5
    verbose = 1
    additional_training_data = True

    print('Loading data...')
    X_train, X_val, y_train, y_val = load_data(test_size, additional_training_data=additional_training_data)

    print('Building model...')
    model = build_model()

    checkpoint = ModelCheckpoint('model_checkpoints/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    print('Compiling model...')
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

    print('Training model...')
    history_object = model.fit_generator(batch_generator(X_train, y_train, batch_size, correction),
                                         steps_per_epoch=len(X_train)/batch_size,
                                         validation_data=batch_generator(X_val, y_val, batch_size, correction),
                                         validation_steps=len(X_val)/batch_size,
                                         callbacks=[checkpoint],
                                         epochs=epochs,
                                         verbose=verbose)

    print('Saving model...')
    model.save('model.h5')

    print('Model saved, training complete!')

    # summarize history for loss
    plt.subplot(111)
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('history.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
