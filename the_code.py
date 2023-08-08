# Import necessary libraries
import mediapipe as mp
import cv2
import numpy as np
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

# Define global variables and initial values
state = True  # On/Off state
valstate = True  # Value On/Off state

global volume
volume = None
volmax = 1.0  # Maximum volume
volmin = 0.0  # Minimum volume

distmax = 150.0  # Maximum distance
distmin = 0  # Minimum distance

# Function to control audio volume
def volumecontrol(volume):
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume_interface = session._ctl.QueryInterface(ISimpleAudioVolume)
        volume_interface.SetMasterVolume(volume, None)

# Function to calculate distance between two points
def distCal(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to check if fingers are touching
def are_fingers_touching(thumb_tip, finger_tip, threshold=30):
    distance = distCal(thumb_tip, finger_tip)
    return distance < threshold

# Function to map volume based on distance
def mapvol(dist):
    prod1 = volmax * dist
    xvol = prod1 / distmax
    
    if xvol >= 1.0:
        xvol = 1.0
    elif xvol <= 0.18:
        xvol = 0.0

    return xvol

# Function to map volume as a percentage value
def mapvoltxt(dist):
    int(dist)
    prod1 = 100 * dist
    xvolc = prod1 / distmax
    
    if xvolc >= 100:
        xvolc = 100
    elif xvolc <= 18:
        xvolc = 0
    return str(int(xvolc)) + str("%")

# Initialize Mediapipe libraries for hand detection and drawing
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize video capture from camera
cap = cv2.VideoCapture(0)

# Start hand detection
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert frame color format and perform horizontal flip
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process hand detection results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks on frame
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )
                # Get positions of interest points on fingers
                thumb_tip = (int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image.shape[1]),
                             int(hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image.shape[0]))
                index_finger_tip = (int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]),
                                    int(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]))
                pinky_tip = (int(hand.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image.shape[1]),
                             int(hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image.shape[0]))

                # Display distance, state, and volume control information on frame
                cv2.putText(image, mapvoltxt(distCal(thumb_tip, index_finger_tip)) + " " + str(state), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Check if fingers are touching and update state
                if are_fingers_touching(thumb_tip, pinky_tip) == True:
                    valstate = not valstate
                elif are_fingers_touching(thumb_tip, pinky_tip) == False:
                    valstate = False

                if valstate == True:
                    state = not state

                # Control volume based on current state
                if state == True:
                    print("On")
                    volumecontrol(mapvol(distCal(thumb_tip, index_finger_tip)))
                elif state == False:
                    print("Off")

        # Display frame with hand detection information
        cv2.imshow('Hand Tracking', image)

        # End video capture by pressing 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
