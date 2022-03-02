# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:08:25 2022

@author: aysenurozkan
"""

import mediapipe as mp
import cv2


class HandDetector():

    def __init__(self, mode=False, maxHands=2, complexity=1,
                 detection_confidence=0.5, tracking_confidence=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.complexity,
                                         self.detection_confidence, self.tracking_confidence)
        self.keyPoints = [4, 8, 12, 16, 20]


    def findHands(self, video, draw=True):
        
        bgr_to_rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(bgr_to_rgb)
        try:
            for landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(video, landmark, self.mp_hands.HAND_CONNECTIONS,
                                                   self.mp_drawing.DrawingSpec(color=(152, 182, 218), thickness=2, circle_radius=2),
                                                   self.mp_drawing.DrawingSpec(color=(31, 18, 193), thickness=2, circle_radius=2))
        except:
            pass

        return video

    def findPosition(self, video):
        # We will keep the landmark points in this list.
        self.lmList = []
        if self.results.multi_hand_landmarks:
            # This is the right hand. 
            hand = self.results.multi_hand_landmarks[0]

            for id, lm in enumerate(hand.landmark):
                height, width, _ = video.shape
                # cx and cy are the position of each landmark
                cx, cy = int(lm.x * width), int(lm.y * height)
                self.lmList.append([id, cx, cy])

        return self.lmList

    def countFinger(self, video):
        count_finger = []
        # Check the thumb is open relative to the x-axis.
        if self.lmList[self.keyPoints[0]][1] > self.lmList[self.keyPoints[0] - 2][1]:
            count_finger.append(1)
        else:
            count_finger.append(0)

        for i in range(1, 5):
            # Check the other fingers are open relative to the y-axis.
            if self.lmList[self.keyPoints[i]][2] < self.lmList[self.keyPoints[i] - 2][2]:
                count_finger.append(1)
            else:
                count_finger.append(0)

        number = count_finger.count(1)

        cv2.rectangle(video, (20, 20), (130, 130), (103, 233, 9), cv2.FILLED)
        cv2.putText(video, str(number), (55, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)


def main():

    cap = cv2.VideoCapture(0)

    detector = HandDetector()
    while cap.isOpened():

        _, video = cap.read()

        video = detector.findHands(video)

        lmList1 = detector.findPosition(video)
        if len(lmList1) != 0:
            detector.countFinger(video)

        cv2.imshow('Video', video)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
