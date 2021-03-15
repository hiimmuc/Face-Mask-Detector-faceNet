# import the necessary packages
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS, VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class FaceNet(object):
    def __init__(self, prototxt_path, weights_path, model_path):
        super(FaceNet, self).__init__()
        self.prototxt_path = prototxt_path
        self.weights_path = weights_path
        self.model_path = model_path

    def creat_net(self):
        t = time.time()
        print("[INFO] read face net")
        self.face_net = cv2.dnn.readNet(self.prototxt_path, self.weights_path)
        print("[INFO] load model...")
        self.mask_net = load_model(self.model_path)
        print("[INFO] done loading!")
        print(time.time() - t)
        # return self.face_net, self.mask_net

    def detector(self, frame):
        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = self.mask_net.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        self.count = [0, 0]
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            condition = mask > withoutMask
            label = "Co deo khau trang" if condition else "Chua deo khau trang"
            color = (0, 255, 0) if condition else (0, 0, 255)
            if mask > withoutMask:
                self.count[0] += 1
            else:
                self.count[1] += 1
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        return frame, self.count

    def video_test(self):
        # initialize the video stream

        print("[INFO] starting video stream...")

        stream = cv2.VideoCapture(0)
        fps = FPS().start()

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            _, frame = stream.read()
            frame = imutils.resize(frame, 640, 360)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            frame, _ = self.detector(frame)

            # loop over the detected face locations and their corresponding
            # locations

            # show the output frame
            cv2.imshow("Frame", frame)
            fps.update()

            # if the `q` key was pressed, break from the loop
            if cv2.waitKey(1) == 27:
                break

        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # do a bit of cleanup
        stream.release()
        cv2.destroyAllWindows()


# # load our serialized face detector model from disk
# prototxtPath = r"backups\deploy.prototxt"
# weightsPath = r"F:backups\res10_300x300_ssd_iter_140000.caffemodel"
# # load the face mask detector model from disk
# model_path = r"F:backups\mask_detector.h5"

# model = FaceNet(prototxt_path=prototxtPath, weights_path=weightsPath, model_path=model_path)
# model.creat_net()
# model.video_test()
