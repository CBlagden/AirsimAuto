from utils import get_safe_tf_session
from keras.backend.tensorflow_backend import set_session
set_session(get_safe_tf_session())

from keras.models import load_model
import cv2
import numpy as np
from pynput.keyboard import Listener
import airsim
from queue import Queue, Empty
import time

controls_queue = Queue()


def on_press(key):
    if key.char == 'r' or key.char == 'e':
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
    time_length = 120

    car_controls = airsim.CarControls()
    car_controls.throttle = 0.6

    client.setCarControls(car_controls)

    i = 0
    start_time = 0
    has_stopped = False
    while i < time_length // delta_time:

        try:
            controls_queue.get_nowait()
        except Empty:
            pass
        else:
            is_api = not is_api
        
        client.enableApiControl(is_api)

        vel = client.getCarState().speed
        print("Vel %f" % vel)

        if is_api:

            response = client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])[0]
            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img.reshape(response.height, response.width, 4)[..., :3]
            img = cv2.resize(img, (64, 64))
            img = np.expand_dims(img, axis=0)

            steering_angle = float(model.predict(img))

            if abs(vel) < 0.1 and not has_stopped:
                start_time = time.time()
                has_stopped = True

            if time.time() - start_time < 0.75 and has_stopped:
                print("Stopped! Heading backwards... {0}".format(time.time() - start_time))
                car_controls.is_manual_gear = True
                car_controls.manual_gear = -1   
                throttle = -1.0
                steering_angle *= -2.0
            else:
                has_stopped = False
                if vel < 10.0:
                    throttle = 0.8
                else:
                    throttle = 0.0
                car_controls.is_manual_gear = False

            car_controls.throttle = throttle
            car_controls.steering = steering_angle / 2.0
            client.setCarControls(car_controls)
    
        time.sleep(delta_time)
        i += 1
