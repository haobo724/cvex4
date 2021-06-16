import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.7, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None
        self.tm_window_size=tm_window_size
        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

	# ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):
        if self.reference is None:
            result = self.detect_face(image)

            return  result

        else:
            x, y, w, h = self.reference["rect"]
            print(x, y, w, h)
            newregion = [x - self.tm_window_size,
                         y - self.tm_window_size,
                         w + self.tm_window_size * 2,
                         h + self.tm_window_size * 2]
            x_new, y_new, _, _ = newregion
            img_new = self.crop_face(image, newregion)
            print(self.reference["aligned"].shape)
            t = cv2.resize(self.reference["aligned"], dsize=(w, h))

            tmp = cv2.matchTemplate(img_new, self.reference["aligned"], method=cv2.TM_CCOEFF_NORMED)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(tmp)
            print(max_val)
            if max_val < self.tm_threshold:
                print("Can not find restart detect face")
                return self.detect_face(image)

            else:
                # face_rect = [x_new + max_loc[0]-self.tm_window_size*2, y_new + max_loc[1]-self.tm_window_size*2,
                #              w+self.tm_window_size, h+self.tm_window_size]
                face_rect = [x_new + max_loc[0] - self.tm_window_size , y_new + max_loc[1] - self.tm_window_size ,
                             w , h ]
                aligned = self.align_face(image, face_rect)
                # cv2.rectangle(image, (a, b), (a + w, b + h), (0, 0, 225), 2)
                # # cv2.rectangle(image, min_loc, (x, y), (0, 0, 225), 2)
                # cv2.imshow("aa", image)
                # cv2.waitKey()

                return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)


        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

