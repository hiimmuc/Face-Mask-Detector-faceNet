from threading import Thread

import cv2


class ThreadedCamera(object):
    def __init__(self, source=0):

        self.capture = cv2.VideoCapture(source)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None


if __name__ == '__main__':
    # stream_link = (0, cv2.CAP_DSHOW)
    streamer = ThreadedCamera()

    while True:
        frame = streamer.grab_frame()
        if frame is not None:
            cv2.imshow("Context", frame)
        cv2.waitKey(1)
