"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import time
import math
import csv
from keras.models import load_model
import os
import contextlib
import io

# Setting recording variables
is_recording = False
last_record_time = 0
record_interval = 0.05  # Record 20 images per second
total_count = 0

display = None
speedometer_image = None


def initialize_display(robot):
    global display, speedometer_image
    display = robot.getDevice("display")
    speedometer_image = display.imageLoad("speedometer.png")


# Create data directory if it doesn't exist
data_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(data_dir, exist_ok=True)

# Initialize CSV file
csv_file_path = os.path.join(data_dir, 'recorded_data.csv')
csv_file = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    ['filename', 'gps_values', 'gyro_values', 'speed', 'angle', 'lidar'])
csv_writer.writerow(['filename', 'angle', 'speed'])

# Save image and log data


def save_image_and_log_data(camera, gps_formatted, gyro_formatted, speed_formatted, angle, min_distance_ahead):
    global last_record_time, csv_writer, csv_file, total_count

    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")
    file_name = current_datetime + ".png"
    file_path = os.path.join(data_dir, file_name)

    # Save image
    camera.saveImage(file_path, 1)
    total_count += 1
    print(f"Image saved: {file_name}", "speed ", speed_formatted,
          "angle ", angle_formatted, "min_distance_ahead ", min_distance_ahead, total_count)

    # Log data to CSV
    # csv_writer.writerow([file_name, gps_formatted,gyro_formatted, speed_formatted, angle_formatted, min_distance_ahead])
    csv_writer.writerow([file_name, "{:.6f}".format(angle), speed_formatted])
    csv_file.flush()
    last_record_time = time.time()

# Getting image from camera


def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))
    return image

# Image processing


def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

# Display image

    """ def display_image(display, image): """
    # Image to display
    image_rgb = np.dstack((image, image, image,))
    # Display image
    image_ref = display.imageNew(image_rgb.tobytes(
    ), Display.RGB, width=image_rgb.shape[1], height=image_rgb.shape[0])
    display.imagePaste(image_ref, 0, 0, False)


# Initial angle and speed
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 0

# Set target speed


def set_speed(kmh):
    global speed
    speed = kmh


def brake():
    global speed
    speed = 0
    print("Braking, speed set to 0")

# Update steering angle


def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle

    # Limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # Update steering angle
    angle = wheel_angle

# Validate increment of steering angle


def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle), turn))


def straight():
    global angle, steering_angle, manual_steering
    angle = 0.0
    steering_angle = 0.0
    manual_steering = 0.0

# Main


def process_lidar_data(lidar):
    # Get the range image (distance values)
    lidar_data = lidar.getRangeImage()

    # Number of points and field of view
    number_of_points = len(lidar_data)
    horizontal_fov = lidar.getFov()  # Horizontal FOV in radians

    # Calculate angles for each distance value
    lidar_angles = np.linspace(-horizontal_fov / 2,
                               horizontal_fov / 2, number_of_points)

    return lidar_data, lidar_angles


def detect_ahead_object(lidar_data, lidar_angles, ahead_angle_range=10.0):
    ahead_angle_range_rad = math.radians(ahead_angle_range)
    min_distance = float('inf')

    for distance, angle in zip(lidar_data, lidar_angles):
        # Check if the angle is within the ahead range
        if -ahead_angle_range_rad <= angle <= ahead_angle_range_rad:
            if distance < min_distance:
                min_distance = distance

    return min_distance


def update_display(gps_values, speed):
    const_needle_length = 50.0

    # Display background
    display.imagePaste(speedometer_image, 0, 0, False)

    # Draw speedometer needle
    current_speed = speed
    alpha = current_speed / 260.0 * 3.72 - 0.27
    x = int(-const_needle_length * math.cos(alpha))
    y = int(-const_needle_length * math.sin(alpha))
    display.drawLine(100, 95, 100 + x, 95 + y)

    # Draw text
    txt = "GPS coords: {:.1f} {:.1f}".format(gps_values[0], gps_values[2])
    display.drawText(txt, 10, 130)
    txt = "GPS speed: {:.1f}".format(speed)
    display.drawText(txt, 10, 140)


###  Usando MODELO  ###

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img[150:230, :]  # Crop the image
    img = cv2.resize(img, (320, 80))  # Resize the image to 320x80
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    # print("Processed Image shape: ", img.shape)
    return img


model = load_model('Grises_model_ok.keras', safe_mode=False)


def main():
    global gps_formatted, gyro_formatted, speed_formatted, angle_formatted, is_recording, last_record_time, min_distance_ahead, camera, angle, previous_distance, speed
    MAX_SPEED = 15
    previous_distance = float('inf')
    speed = MAX_SPEED
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Initialize the display
    initialize_display(robot)

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Create gps instance
    gps = robot.getDevice("gps")
    gps.enable(timestep)

    # Create Gyro instance
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    # Create Lidar instance
    lidar = robot.getDevice("lidar")
    lidar.enable(timestep)

    # Create keyboard instance
    keyboard = Keyboard()
    keyboard.enable(timestep)

    def format_values(gps_values, gyro_values, speed, angle):
        gps_formatted = ["{:.2f}".format(val) for val in gps_values]
        gyro_formatted = ["{:9.5f}".format(val) for val in gyro_values]
        speed_formatted = "{:.2f}".format(speed)
        angle_formatted = "{:.2f}".format(angle)
        return gps_formatted, gyro_formatted, speed_formatted, angle_formatted

    def process_lidar_data(lidar):
        lidar_data = lidar.getRangeImage()
        number_of_points = len(lidar_data)
        lidar_angles = np.linspace(-lidar.getFov() / 2,
                                   lidar.getFov() / 2, number_of_points)
        x_points = []
        y_points = []

        for i in range(number_of_points):
            distance = lidar_data[i]
            angle = lidar_angles[i]
            if distance < lidar.getMaxRange():
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                x_points.append(x)
                y_points.append(y)

        return x_points, y_points

    # Delay for keys
    last_key_press_time = 0
    key_press_interval = 0.05

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        y_pred = model.predict(preprocess_image(image), verbose=0)
        predicted_angle = y_pred[0][0] / 10
        angle = predicted_angle

        # Get gps values
        gps_values = gps.getValues()

        # Get gyro values
        gyro_values = gyro.getValues()

        # Get Lidar values
        lidar_data, lidar_angles = process_lidar_data(lidar)

        # Detect the closest object ahead
        min_distance_ahead = detect_ahead_object(lidar_data, lidar_angles)

        gps_formatted, gyro_formatted, speed_formatted, angle_formatted = format_values(
            gps_values, gyro_values, speed, angle)

        speed_change_message = ""
        if min_distance_ahead < 15.0:
            if min_distance_ahead < previous_distance:

                if min_distance_ahead > 1.0:
                    speed = max(speed - 1.0, min_distance_ahead)
                else:
                    speed = 0
                speed_change_message = "\treducing speed"
            else:
                speed_change_message = "\tkeeping speed"
        else:
            # Increase speed but keep it as the distance to the nearest object
            speed = min(speed + 1.0, MAX_SPEED, min_distance_ahead)
            speed_change_message = "\tspeeding"

        previous_distance = min_distance_ahead

        if speed == MAX_SPEED:
            speed_change_message = ""

        print("Predicted angle: ", "{:.4f}".format(angle)+"\trad", "   speed: ",
              str(speed)+" k/h", "\tObject ahead: ", "{:.4f}".format(min_distance_ahead)+" m" + speed_change_message)
        current_time = time.time()
        if current_time - last_key_press_time > key_press_interval:
            key = keyboard.getKey()
            if key == keyboard.UP:  # up
                set_speed(speed + 1.0)
                driver.setCruisingSpeed(speed)  # Update the cruising speed
                print("up", speed)
            elif key == keyboard.DOWN:  # down
                set_speed(speed - 1.0)
                driver.setCruisingSpeed(speed)  # Update the cruising speed
                print("down", speed)
            elif key == keyboard.RIGHT:  # right
                change_steer_angle(+.2)
                print("right")
            elif key == keyboard.LEFT:  # left
                change_steer_angle(-.2)
                print("left")
            elif key == 32:  # spacebar for braking
                brake()
            elif key == ord('C'):
                straight()
            elif key == ord('A'):
                is_recording = not is_recording
                print("Recording started" if is_recording else "Recording stopped")

            last_key_press_time = current_time

        if is_recording and (current_time - last_record_time > record_interval):
            save_image_and_log_data(camera, gps_formatted, gyro_formatted,
                                    speed_formatted, angle, min_distance_ahead)

        # Update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

        # Update the display with the current speed and GPS coordinates
        update_display(gps_values, speed)

    csv_file.close()


if __name__ == "__main__":
    main()
