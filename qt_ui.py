"""
UI Modified from https://github.com/yushulx/python/blob/master/examples/qt/barcode-reader.py
for CS549 group M
"""

import sys
from PyQt5.QtGui import QPixmap, QImage

from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit, \
    QSizePolicy, QMessageBox, QHBoxLayout
from PyQt5.QtCore import Qt, QStringListModel, QSize, QTimer

from detection_controller import DetectionController


class UI_Window(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        # Create a timer.
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)

        # Create a layout.
        layout = QVBoxLayout()

        # Add a button
        self.btn = QPushButton("Load an image")
        self.btn.clicked.connect(self.pickFile)
        layout.addWidget(self.btn)

        # Add a button
        button_layout = QHBoxLayout()

        btnCamera = QPushButton("Open camera")
        btnCamera.clicked.connect(self.openCamera)
        button_layout.addWidget(btnCamera)

        btnCamera = QPushButton("Stop camera")
        btnCamera.clicked.connect(self.stopCamera)
        button_layout.addWidget(btnCamera)

        layout.addLayout(button_layout)

        # Add a label to hold raw image
        self.raw_frame = QLabel()
        self.raw_frame.setFixedSize(640, 360)
        layout.addWidget(self.raw_frame)

        # Add a label to hold filtered image
        self.filtered_frame = QLabel()
        self.filtered_frame.setFixedSize(640, 360)
        layout.addWidget(self.filtered_frame)

        # Set last frame
        self.last_frame = None

        # Create controller
        self.det_controller = DetectionController()

        # Add a text area
        self.results = QTextEdit()
        layout.addWidget(self.results)

        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle("Leprechaun Detector")
        self.setFixedSize(800, 800)

    def closeEvent(self, event):
        msg = "Close the app?"
        reply = QMessageBox.question(self, 'Message',
                                     msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            self.stopCamera()
        else:
            event.ignore()

    def resizeImage(self, filename):
        pixmap = QPixmap(filename)
        lwidth = self.raw_frame.maximumWidth()
        pwidth = pixmap.width()
        lheight = self.raw_frame.maximumHeight()
        pheight = pixmap.height()

        wratio = pwidth * 1.0 / lwidth
        hratio = pheight * 1.0 / lheight

        if pwidth > lwidth or pheight > lheight:
            if wratio > hratio:
                lheight = pheight / wratio
            else:
                lwidth = pwidth / hratio

            scaled_pixmap = pixmap.scaled(lwidth, lheight)
            return scaled_pixmap
        else:
            return pixmap

    def pickFile(self):
        self.stopCamera()
        # Load an image file.
        filename = QFileDialog.getOpenFileName(self, 'Open file',
                                               'E:\\Program Files (x86)\\Dynamsoft\\Barcode Reader 7.1\\Images',
                                               "Barcode images (*)")
        # Show barcode images
        pixmap = self.resizeImage(filename[0])
        self.raw_frame.setPixmap(pixmap)

        # Read barcodes
        self.readBarcode(filename[0])

    def openCamera(self):
        self.timer.start(1000. / 24)

    def stopCamera(self):
        self.timer.stop()

    def getPos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.det_controller.handle_click(x, y)

        # https://stackoverflow.com/questions/41103148/capture-webcam-video-using-pyqt
    def nextFrameSlot(self):
        # Get frames
        raw, filtered = self.det_controller.process_frame()

        # Convert to QImage
        raw_image = QImage(raw, raw.shape[1], raw.shape[0], QImage.Format_RGB888)
        filtered_image = QImage(filtered, filtered.shape[1], filtered.shape[0], QImage.Format_RGB888)

        # Add to frame as pixmap
        raw_pixmap = QPixmap.fromImage(raw_image)
        filtered_pixmap = QPixmap.fromImage(filtered_image)
        self.raw_frame.setPixmap(raw_pixmap)
        self.filtered_frame.setPixmap(filtered_pixmap)

        # Add click handler
        self.raw_frame.mousePressEvent = self.getPos


def main():
    app = QApplication(sys.argv)
    ex = UI_Window()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()