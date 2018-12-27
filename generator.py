from keras.utils import Sequence
from sklearn.utils import shuffle
import numpy as np
import cv2


class DriveDataGenerator(Sequence):

    def __init__(self,
                 images,
                 labels,
                 resize_dims=(128, 128),
                 batch_size=32,
                 roi=None,
                 horizontal_flip=True,
                 flip_percentage=0.5,
                 zero_drop_percentage=0.7,
                 shuffle=True):
        if roi is None:
            roi = [76, 135, 0, 255]
        self.images = images
        self.labels = labels
        self.resize_dims = resize_dims
        self.batch_size = batch_size
        self.roi = roi
        self.shuffle = shuffle

    def __getitem__(self, index):
        batch_x = self.images[index * self.batch_size:
                              (index + 1) * self.batch_size]
        batch_y = self.labels[index * self.batch_size:
                              (index + 1) * self.batch_size]
        imgs = []
        steering_angles = []
        for i, img in enumerate(batch_x):
            img = img[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=2)
            steering_angle = batch_y[i]

            img, steering_angle = self.random_shear(img, steering_angle, shear_range=100)
            
#             img, steering_angle = self.random_crop(img, steering_angle, tx_lower=-20, tx_upper=20, ty_lower=-10,
#                                                    ty_upper=10)
            img, steering_angle = self.random_flip(img, steering_angle)
            img = self.random_brightness(img)
            imgs.append(img)
            steering_angles.append(steering_angle)

        if self.shuffle:
            imgs, steering_angles = shuffle(imgs, steering_angles)

        return np.array(imgs), np.array(steering_angles)

    def random_crop(self, image, steering=0.0, tx_lower=-20, tx_upper=20, ty_lower=-2, ty_upper=2, rand=True):
        # we will randomly crop subsections of the image and use them as our data set.
        # also the input to the network will need to be cropped, but of course not randomly and centered.
        shape = image.shape
        col_start, col_end = abs(tx_lower), shape[1] - tx_upper
        horizon = 15
        bonnet = 34
        if rand:
            tx = np.random.randint(tx_lower, tx_upper + 1)
            ty = np.random.randint(ty_lower, ty_upper + 1)
        else:
            tx, ty = 0, 0
        
        
        random_crop = image[horizon + ty:bonnet + ty, col_start + tx:col_end + tx, :]
        image = cv2.resize(random_crop, self.resize_dims, cv2.INTER_AREA)
        
        # the steering variable needs to be updated to counteract the shift
        if tx_lower != tx_upper:
            dsteering = -tx / (tx_upper - tx_lower) / 3.0
        else:
            dsteering = 0
        steering += dsteering

        return image, steering

    def random_shear(self, image, steering, shear_range):
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        #    print('dx',dx)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
        steering += dsteering

        return image, steering

    def random_brightness(self, image):
        image1 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
        random_bright = 0.8 + 0.4 * (2 * np.random.uniform() - 1.0)
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image1 = np.expand_dims(image1, axis=2)
        return image1

    def random_flip(self, image, steering):
        coin = np.random.randint(0, 2)
        if coin == 0:
            image, steering = cv2.flip(image, 1), -steering
        return image, steering

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
