from djitellopy import Tello
import cv2
from time import sleep

global yaw_PID, ud_PID

yaw_PID = {'kp': 0.25, 'ki': 0.01, 'kd': 0.7, 'prev_err': 0,
           'max_int': 10, 'max_output': 50, 'min_output': -50}

ud_PID = {'kp': 0.3, 'ki': 0.01, 'kd': 0.7, 'prev_err': 0,
          'max_int': 10, 'max_output': 50, 'min_output': -50}

fb_PID = {'kp': 0.3, 'ki': 0.02, 'kd': 0.7, 'prev_err': 0,
          'max_int': 15, 'max_output': 50, 'min_output': -50}


def drone_init():
    '''
    Function to Init the Drone and return the Tello object
    '''

    drone = Tello()
    sleep(1)
    drone.connect()
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0

    battery_percent = drone.get_battery()
    print("Battery: " + str(battery_percent))

    drone.streamon()
    return drone


def tello_get_frame(drone, w=640, h=480):
    '''
    Function to simplify getting the frame from the drone
    '''

    img = drone.get_frame_read()
    img = img.frame
    img = cv2.resize(img, (w, h))

    return img


def track_person(drone, ud_PID, yaw_PID, current, setpoint, lrVel, fbVel):

    # Calculate Yaw error
    err = setpoint[0] - current[0]

    yaw_PID_val = yaw_PID['kp'] * err                            # Proportional
    yaw_PID_val += yaw_PID['kd'] * (err - yaw_PID['prev_err'])   # Derivative

    yaw_int = yaw_PID['ki'] * (err + yaw_PID['prev_err'])

    # Limit the yaw_PID_val
    yaw_int = min(yaw_PID['max_int'], max(-yaw_PID['max_int'], yaw_int))

    yaw_PID_val += yaw_int                                      # Integral

    # Update previous error
    yaw_PID['prev_err'] = err

    # Limit PID output
    yaw_PID_val = max(
        min(yaw_PID_val, yaw_PID['max_output']), yaw_PID['min_output'])
    yaw_PID_val = int(yaw_PID_val)

    # Same Procedure for the Vertical error
    err = setpoint[1] - current[1]
    # Proportional
    ud_PID_val = ud_PID['kp'] * err
    ud_PID_val += ud_PID['kd'] * \
        (err - ud_PID['prev_err'])                 # Derivative

    ud_int = ud_PID['ki'] * (err + ud_PID['prev_err'])

    # Limit the ud_PID_val
    ud_int = min(ud_PID['max_int'], max(-ud_PID['max_int'], ud_int))

    # Integral
    ud_PID_val += ud_int

    # Update previous error
    ud_PID['prev_err'] = err

    # Limit PID output
    ud_PID_val = max(
        min(ud_PID_val, ud_PID['max_output']), ud_PID['min_output'])
    ud_PID_val = int(ud_PID_val)

    # Same Procedure for the Forward and Backwards error
    err = setpoint[2] - current[2]
    # Proportional
    fb_PID_val = fb_PID['kp'] * err
    fb_PID_val += fb_PID['kd'] * \
        (err - fb_PID['prev_err'])                 # Derivative

    fb_int = fb_PID['ki'] * (err + fb_PID['prev_err'])

    # Limit the fb_PID_val
    fb_int = min(fb_PID['max_int'], max(-fb_PID['max_int'], fb_int))

    # Integral
    fb_PID_val += fb_int

    # Update previous error
    fb_PID['prev_err'] = err

    # Limit PID output
    fb_PID_val = max(
        min(fb_PID_val, fb_PID['max_output']), fb_PID['min_output'])
    fb_PID_val = int(fb_PID_val)

    # Update drone yaw
    drone.send_rc_control(int(lrVel), fb_PID_val, ud_PID_val, yaw_PID_val)

    return yaw_PID_val, ud_PID_val, fb_PID_val
