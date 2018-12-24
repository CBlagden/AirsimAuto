import numpy as np
from keras.utils import Sequence


class DriveDataGenerator(Sequence):

    def __init__(self, images, labels,
                 num_images_input=4,
                 batch_size=32):
        self.images = images
        self.labels = labels
        assert len(images) == len(labels)

        self.num_images_input = num_images_input
        self.batch_size = batch_size * num_images_input

    def __getitem__(self, index):
        batch_x = self.images[index * self.batch_size:
                              (index + 1) * self.batch_size]
        batch_y = self.labels[index * self.batch_size:
                              (index + 1) * self.batch_size]

        images = []
        labels = []
        for i in range(0, self.batch_size, self.num_images_input):
            images.append(np.concatenate([batch_x[i + j] for j in range(self.num_images_input)],
                                         axis=2))
            labels.append(batch_y[i + self.num_images_input - 1])

        return np.array(images), np.array(labels)

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
