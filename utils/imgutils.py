import math
import cv2
import numpy as np
from time import time, sleep
import mediapipe as mp
import os
import datetime


def print_pose_details(results, image_width, image_height):
    '''
    Function to Print the Landmark Poses
    '''
    if results.pose_landmarks:
        for i in range(len(results.pose_landmarks.landmark)):
            prompt = f'''
            Land Mark Name: {mp.solutions.pose.PoseLandmark(i).name}:
                x: {results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark(i).value].x * image_width}
                y: {results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark(i).value].y * image_height}
                z: {results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark(i).value].z * image_width}
                visibility: {results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark(i).value].visibility}\n
            '''

            print(prompt)


def landmark_image_prepare(image, pose, image_width, image_height, draw_landmark=True, flip_image=False):
    '''
    Function to DeNormalize the Landmark Poses Based on Mediapipe Documentation and Annotate the Image
    '''

    image = cv2.resize(image, (image_width, image_height))

    if flip_image:
        image = cv2.flip(image, 1)

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if draw_landmark:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks is not None:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x * image_width,
                              landmark.y * image_height,
                              landmark.z * image_width))

    return image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    Function to Calculate the Angle of a Joint
    '''
    l1 = np.array([landmark1[0], landmark1[1]])
    l2 = np.array([landmark2[0], landmark2[1]])
    l3 = np.array([landmark3[0], landmark3[1]])

    v1 = l1 - l2
    v2 = l3 - l2

    angle = np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    angle = angle * 180 / math.pi

    return angle


def gimmeAngles(landmarks, Print=False):
    '''
    Function to Calculate the Angles of the Joints
    '''

    if len(landmarks) == 0:
        return None

    else:
        right_elbow_angle = calculateAngle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                                           landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value])

        left_elbow_angle = calculateAngle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
                                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value])

        right_shoulder_angle = calculateAngle(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value])

        left_shoulder_angle = calculateAngle(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value],
                                             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                                             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value])

        angles = (left_elbow_angle, right_elbow_angle,
                  left_shoulder_angle, right_shoulder_angle)

        if Print:
            prompt = f'L E: {left_elbow_angle}, R E: {right_elbow_angle}, L S: {left_shoulder_angle}, R S: {right_shoulder_angle}'
            print(prompt)

        return angles


def isinrange(angle, origin, threshold):
    '''
    Function to Check if the Angle is in the Range of the Origin
    '''

    min = origin - threshold
    max = origin + threshold

    if angle > min and angle < max:
        return True

    return False


# Pose Classification Tree
def classifyPose(angles, threshold=15):
    '''
    Function to Detect the Pose Based on the Angles

    Poses: 
        1. Flip
        2. Snapshot ==> Snap
        3. Back
        4. For
        5. Left
        6. Right
        7. Land
        0. Else: Active Track ==> AT
    '''

    if angles is None:
        return 'AT'

    else:
        (LE, RE, LS, RS) = angles

        # Flip Pose
        if isinrange(LE, 90, threshold) and isinrange(RE, 90, threshold) and isinrange(LS, 90, threshold) and isinrange(RS, 90, threshold):
            return 'Flip'

        # SnapShot Pose
        elif isinrange(LE, 0, threshold) and isinrange(RE, 0, threshold) and isinrange(LS, 0, threshold) and isinrange(RS, 0, threshold):
            return 'Snap'

        # Back Pose
        elif isinrange(LE, 90, threshold) and isinrange(RE, 180, threshold) and isinrange(LS, 90, threshold) and isinrange(RS, 90, threshold):
            return 'Back'

        # For Pose
        elif isinrange(LE, 180, threshold) and isinrange(RE, 90, threshold) and isinrange(LS, 90, threshold) and isinrange(RS, 90, threshold):
            return 'For'

        # Left Pose
        elif isinrange(LE, 180, threshold) and isinrange(RE, 180, threshold) and isinrange(LS, 90, threshold) and isinrange(RS, 0, threshold):
            return 'Left'

        # Right Pose
        elif isinrange(LE, 180, threshold) and isinrange(RE, 180, threshold) and isinrange(LS, 0, threshold) and isinrange(RS, 90, threshold):
            return 'Right'

        # Land Pose
        elif isinrange(LE, 180, threshold) and isinrange(RE, 180, threshold) and isinrange(LS, 180, threshold) and isinrange(RS, 180, threshold):
            return 'Land'

        else:
            return 'AT'



# Function to take a SnapShot and Save it to the Directory with Specified Image Name
def snapshot(image):
    '''
    Function to Save the Image Into the Repo Named Snapshots 
    '''
    # Create the Repo if it Does not Exist
    os.makedirs('./Snapshots', exist_ok=True)

    # Create the File Name
    file_name = './Snapshots/' + str(datetime.datetime.now()) + '.jpg'

    # Save the Image
    cv2.imwrite(file_name, image)
