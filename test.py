from keras.models import load_model
import airsim
import cv2
import numpy as np
import time
import keyboard


if __name__ == '__main__':
    model = load_model('model.h5')

    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)


    num_images_input = 4

    print("Enabled API")

    # seconds
    delta_time = 0.1
    time_length = 30

    car_controls = airsim.CarControls()
    car_controls.throttle = 0.3
    client.setCarControls(car_controls)

    i = 0
    while i < time_length//delta_time:

        response = client.simGetImages(
            [airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 4)[..., :3]
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(np.expand_dims(img, axis=2), axis=0)

        steering_angle = float(model.predict(img))
        car_controls.throttle = 0.5
        car_controls.steering = steering_angle

        client.setCarControls(car_controls)

        time.sleep(delta_time)
        i += 1
