import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.models import load_model
import cv2
import numpy as np
from pynput.keyboard import Listener
import airsim
from queue import Queue, Empty
import time

controls_queue = Queue()


def on_press(key):
    if key.char == 'r':
        controls_queue.put(key)


if __name__ == '__main__':

    listener = Listener(on_press=on_press)
    listener.start()

    model = load_model('model.h5')

    is_api = True

    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(is_api)

    # seconds
    delta_time = 0.05
    time_length = 30

    car_controls = airsim.CarControls()
    car_controls.throttle = 0.0

    client.setCarControls(car_controls)

    i = 0
    while i < time_length // delta_time:

        try:
            controls_queue.get_nowait()
        except Empty:
            pass
        else:
            is_api = not is_api
        
        client.enableApiControl(is_api)

        if is_api:
            response = client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])[0]
            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img.reshape(response.height, response.width, 4)[..., :3]
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(np.expand_dims(img, axis=2), axis=0)

            steering_angle = float(model.predict(img))
            car_controls.throttle = 0.5
            car_controls.steering_angle = steering_angle
            client.setCarControls(car_controls)
    
        time.sleep(delta_time)
        i += 1
