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
                cv2.circle(frame, tuple(ld[0]), 5, (0, 255, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 5, (0, 255, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 255, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 255, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 255, 255), -1)
        except Exception as e:
            pass
    

    def run(self):
        # Capturing the webcam
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            try:
                # Detect face box, probability and facial landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

                # Draw on frame
                self.draw(frame, boxes, probs, landmarks)

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