import cv2
import numpy as np
from normal_pixel_distribution import ObjectSample


class cv_ui:
    def __init__(self, sample_file='sample.npy'):
        self.frame = None
        self.frame_hsv = None
        self.sample = ObjectSample(sample_file)
        cv2.namedWindow('Webcam')
        cv2.setMouseCallback('Webcam', self.click_point)

    def click_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.sample.add_data(self.frame_hsv[y, x])
            self.sample.calculate_stats()
            self.sample.save()
            pdf = self.sample.process_image(self.frame_hsv)
            print(x, y, self.frame_hsv[y, x], pdf[y, x], self.sample.process_pixel(self.frame_hsv[y, x]))

    def gradient_webcam(self):
        cap = cv2.VideoCapture(0)
        key = 0
        while key != ord('q'):
            ret, raw = cap.read()
            self.frame = cv2.resize(raw, (640, 360))
            self.frame_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            cv2.setMouseCallback('Webcam', self.click_point)

            pdf = self.sample.process_image(self.frame_hsv)
            pdf_gray = cv2.cvtColor(pdf, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Webcam", np.vstack((self.frame, pdf_gray)))
            key = cv2.waitKey(1)

            if key == ord('s'):  # Save
                self.sample.save()



        cap.release()
        cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    ui = cv_ui()
    ui.gradient_webcam()
