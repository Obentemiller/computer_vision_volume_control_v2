Hand tracking in MediaPipe is based on Convolutional Neural Networks (CNNs) and has been trained on a large dataset of hands in various positions and orientations. It operates in real-time and is capable of detecting and tracking hands in real-time image streams.

How it works:

Initial Hand Detection: The first step is hand detection in the image. The neural network is used to locate regions that likely contain hands.

Landmark Points: After detecting a hand, the neural network identifies several landmark points present in the capture, representing different parts of the hand such as fingers, joints, and the palm. Typically, around 21 landmark points are detected per hand.

Position Estimation: Based on the identified landmark points, the algorithm estimates the 3D position, orientation, and shape of the hand in the image space, all in real-time.

All landmark point determinations are detailed in this image:
<img width="1073" alt="hand-landmarks" src="https://github.com/Obentemiller/computer_vision_volume_control_v2/assets/135489040/da0f9223-2b06-418a-ad9f-03c7c2d9cb7a">
