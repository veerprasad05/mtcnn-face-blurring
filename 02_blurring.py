from facenet_pytorch import MTCNN
import cv2
import numpy as np
import torch


class FaceDetector(object):
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def draw(self, frame, boxes, probs, landmarks):
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                box = box.astype('int')
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=2)

                # Show probability
                cv2.putText(frame, str(round(prob, 4)), (((box[0] + box[2]) >> 1) - 20, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Draw landmarks
                ld = ld.astype('int')
                cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except Exception as e:
            pass
    
    def _detect_ROIs(self, boxes):
        """
        Return ROIs as a list
        (X1,X2,Y1,Y2)
        """
        ROIs = list()
        for box in boxes:
            ROI = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
            ROIs.append(ROI)

        return ROIs

    def _blur_face(self, image, factor=2.0):
        """
        Return the blured image
        """
        # Determine size of blurring kernel based on input image
        (h,w) = image.shape[:2]
        kW = int(w/factor)
        kH = int(h/factor)

        # Ensure width and height of kernel are odd
        if kW % 2 == 0:
            kW -= 1

        if kH % 2 == 0:
            kH -= 1

        # Apply a Gaussian blur to the input image using our computed kernel size
        return cv2.GaussianBlur(image, (kW, kH), 0)

    def run(self):
        # Capturing the webcam
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            try:
                # Detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(
                    frame, landmarks=True)
                # Draw on frame
                #self.draw(frame, boxes, probs, landmarks)

                # Extract the face ROI
                ROIs = self._detect_ROIs(boxes)
                for roi in ROIs:
                    (startY,endY, startX,endX) = roi
                    face = frame[startY:endY, startX:endX]
                    blured_face = self._blur_face(face)
                    frame[startY:endY, startX:endX] = blured_face

                # Show the frame
                cv2.imshow('Face Detection', frame)

            except Exception as e:
                pass

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


mtcnn = MTCNN()
fd = FaceDetector(mtcnn)
fd.run()