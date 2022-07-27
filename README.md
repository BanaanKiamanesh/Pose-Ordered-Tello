# **Pose Ordered Tello**

Human-Robot-Interaction using Pose-Estimation tree.

---

### **The code contains of 3 main parts:**

1. Pose Estimation
2. Body Tracking
3. And some other actions for Sending Commands for the Tello Drone!

### **The Commands Part Contains of:**

1. Active Track
2. Forward Movement
3. Backward Movement
4. Left Movement
5. Right Movement
6. Flipping Backward
7. Taking a Snapshot
8. Land

![Visual Explanation](images/Tello.jpg)

### **For the Active Track Part:**

3 PID Controllers are applied for tracking the person in the frame.

1. To control the altitude(Tracking Error = Y position of the person in each frame).
2. Control the orientation(TE = X position of the person in the frame).
3. Control the distance(TE = height difference of Hip and Nose of the person).

### **For the Pose Detection:**

1. Body joints were detected using Google's Mediapipe Framework
2. Angles of the elbows and shoulders for both right and left hands are being detected
3. Based on the given angles, each pose is getting detected
4. Based on that, the commands are being sent to the Quadcopter
5. After executing each command, 50 frames of images are ignored until the following command arises!

***Note:*** Only tested on Ubuntu 20.04 and Windows 11(Perfectly Working!)!

---
