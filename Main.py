#!/bin/python3

from numpy import uint8
from utils.imgutils import *
from utils.droneutils import *

# Params
POSE_ANGLE_THRESHOLD = 20
MOVE_CONSTANT_DIST = 70
FRAMES_TO_IGNORE = 50

MIN_DIST = 60
MAX_DIST = 280


# Initialize the drone
tello = drone_init()
is_takeoff = False

# FPS Specs
pTime = 0
Time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# get image_width and height
global image_width, image_height
image_width = 640
image_height = 480

frame_count = 0

# Left Right Specs
lrVel = 0
fbVel = 0
distSetpoint = 200


# print("Image width: " + str(image_width))
# print("Image height: " + str(image_height))

with mp.solutions.pose.Pose(
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
        smooth_landmarks=True,
        model_complexity=2) as pose:

    while True:

        # Take off and Rise
        if is_takeoff == False:
            tello.takeoff()
            tello.send_rc_control(0, 0, 50, 0)
            sleep(3)
            tello.send_rc_control(0, 0, 0, 0)

            is_takeoff = True

        if frame_count > 0:
            frame_count -= 1

        # Image Acquisition
        image = tello_get_frame(tello, image_width, image_height)

        # Extract Landmarks of the Frame
        image, landmarks = landmark_image_prepare(
            image, pose, image_width,
            image_height,  draw_landmark=False, flip_image=True)

        # Calculate Angles of the Sholders and Elbows
        angles = gimmeAngles(landmarks=landmarks, Print=False)

        # Classify Pose and Put text int the middle of the Image
        pose_type = classifyPose(angles=angles, threshold=POSE_ANGLE_THRESHOLD)

        if frame_count == 0:

            if pose_type == "AT":
                pass

            elif pose_type == "Flip":
                tello.flip_back()
                frame_count = FRAMES_TO_IGNORE

            elif pose_type == "Right":
                lrVel = -100
                frame_count = FRAMES_TO_IGNORE

            elif pose_type == "Left":
                lrVel = 100
                frame_count = FRAMES_TO_IGNORE

            elif pose_type == "For":
                distSetpoint += 50

                if distSetpoint >= MAX_DIST:
                    distSetpoint = MAX_DIST

                frame_count = FRAMES_TO_IGNORE

            elif pose_type == "Back":
                distSetpoint -= 50

                if distSetpoint <= MIN_DIST:
                    distSetpoint = MIN_DIST

                frame_count = FRAMES_TO_IGNORE

            elif pose_type == "Snap":
                snapshot(image)
                frame_count = FRAMES_TO_IGNORE

            elif pose_type == "Land":
                tello.send_rc_control(0, 0, 0, 0)

                sleep(2)

                tello.land()
                is_takeoff = False

                cv2.destroyAllWindows()
                break

        # Get the Setpoint for the Controllers
        if len(landmarks) > 0:
            nose_x, nose_y, _ = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
            hip_x, hip_y, _ = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            fbError = hip_y - nose_y

        else:  # If there is no person in the frame
            nose_x = image_width // 2
            nose_y = image_height // 2

            fbError = 200

        # Uncomment for to see the Tracking Point and Depth Error

        # image = cv2.circle(image, (int(nose_x), int(nose_y)),
        #                    15, (255, 255, 255), -1)

        # cv2.putText(image, str(int(fbError)), (int(nose_x), int(
        #     nose_y)), font, 0.5, (255, 100, 100), 1, cv2.LINE_AA)

        yaw_correction, ud_correction, fb_PID_val = track_person(tello, ud_PID, yaw_PID, (
            nose_x, nose_y, fbError), (image_width // 2, image_height // 2, distSetpoint), lrVel, fbVel)

        # Get FPS
        Time = time()
        fps = 1 / (Time - pTime + 1e-9)
        pTime = Time
        fps = "FPS = " + str(int(fps))

        # Extend the Image for Drone States
        image = np.concatenate(
            (np.ones((480, 160, 3)).astype(uint8) * 40, image), axis=1)

        # Annotate the Image with States and Specs
        cv2.putText(image, fps, (10, 80), font, 0.7,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Battery: " + str(tello.get_battery()),
                    (10, 30), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Mode: " + pose_type, (10, 130),
                    font, 0.6, (255, 128, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Distance: " + str(distSetpoint), (10, 160),
                    font, 0.5, (255, 128, 0), 1, cv2.LINE_AA)

        cv2.putText(image, "Yaw    : " + str(yaw_correction),
                    (10, 450), font, 0.5, (255, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(image, "Vertical: " + str(ud_correction),
                    (10, 420), font, 0.5, (255, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(image, "Back n Forth: " + str(fb_PID_val),
                    (10, 390), font, 0.5, (255, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow('Tello Stream', image)

        # Lower the Right/Left Movement Speed
        lrVel *= 0.9
        fbVel *= 0.9

        # Emergency Land Button ==> q
        if cv2.waitKey(1) & 0xFF == ord('q'):

            if is_takeoff:

                tello.send_rc_control(0, 0, 0, 0)

                tello.land()
                is_takeoff = False

                cv2.destroyAllWindows()
                sleep(1)
                break

            break

tello.end()
