import time
from threading import Thread

import cv2
# from facedetect_yolo import Yolov4
from imutils.video import FPS


class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class VideoShow:
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == 27:
                self.stopped = True

    def stop(self):
        self.stopped = True


# label = r"backup/obj.names"
# config = r"backup/yolov4-tiny-custom.cfg"
# net_path = r"backup/yolov4-tiny-custom_best.weights"

# print("[INFO] Loading net...")
# t = time.time()
# myYolo = Yolov4(net_path=net_path, config=config, label=label)
# print(f"[INFO] Done in {round(time.time() - t, 2)} s")


# def threadBoth(source=0):
#     video_getter = VideoGet(source).start()
#     video_shower = VideoShow(video_getter.frame).start()
#     fps = FPS().start()
#     delay = 0
#     while True:
#         if video_getter.stopped or video_shower.stopped:
#             video_shower.stop()
#             video_getter.stop()
#             break

#         frame = video_getter.frame
#         output_img, cond = myYolo.detector(frame, 0.3, 0.5, delay)
#         if cond:
#             delay = delay + 1 if delay <= 3 else 0
#         video_shower.frame = output_img
#         fps.update()
#     fps.stop()
#     print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#     print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# threadBoth()
#window_video.py
import sys, cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
"""
Author: Wang Hangzhou
 Reference: https://blog.csdn.net/oscar_liu/article/details/81210301
 Date: 2019-06-08,
 Description: Python language Play video from a USB camera using PyQt5 (eg laptop camera)
         Optimized to minimize cpu utilization, energy saving, environmentally stable, suitable for long-term operation
"""

class Video(QMainWindow):
    def __init__(self, cam):
        super().__init__()

                 # Initialize the incoming camera handle as an instance variable and get the camera width and height
        self.cam = cam
        self.w = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

                 # Set the position and size of the GUI window
        self.setGeometry(300, 200, self.w+200, self.h+200)
        self.setWindowTitle('Zhejiang Radio and Television Group TV Broadcast Center')
                 # Print the width and height of the camera image
        print(self.w, self.h)
        self.vF = QLabel()

        self.setCentralWidget(self.vF)

                 #Set the video to be displayed in the middle of the window, otherwise it can be commented out
        self.vF.setAlignment(Qt.AlignCenter)

                 # Setting the timer Execute the instance's play function every 25 milliseconds to refresh the image.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.play)
        self._timer.start(25)

    def play(self):
        """
                 Get the image from the camera, first convert to RGB format, then generate QImage object.
                 Then use this QImage to refresh the vF instance variable to refresh the video screen.
        """
        r, f = self.cam.read()
        if r:
            self.vF.setPixmap(QPixmap.fromImage(
                QImage(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                       self.w,
                       self.h,
                       13)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
         # Initialize the GUI window and pass in the camera handle
    win = Video(cv2.VideoCapture(0))
    win.show()
    sys.exit(app.exec_())
