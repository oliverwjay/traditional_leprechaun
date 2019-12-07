"""
UI Modified from https://github.com/yushulx/python/blob/master/examples/qt/barcode-reader.py
for CS549 group M
"""

import sys
from PyQt5.QtGui import QPixmap, QImage

from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit, \
    QSizePolicy, QMessageBox, QHBoxLayout
from PyQt5.QtCore import Qt, QStringListModel, QSize, QTimer

import os
import cv2


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

        # Add a label
        self.label = QLabel()
        self.label.setFixedSize(640, 480)
        layout.addWidget(self.label)

        # Set last frame
        self.last_frame = None

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
        lwidth = self.label.maximumWidth()
        pwidth = pixmap.width()
        lheight = self.label.maximumHeight()
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
        self.label.setPixmap(pixmap)

        # Read barcodes
        self.readBarcode(filename[0])

    def openCamera(self):
        self.vc = cv2.VideoCapture(0)
        # vc.set(5, 30)  #set FPS
        self.vc.set(3, 640)  # set width
        self.vc.set(4, 480)  # set height

        if not self.vc.isOpened():
            msgBox = QMessageBox()
            msgBox.setText("Failed to open camera.")
            msgBox.exec_()
            return

        self.timer.start(1000. / 24)

    def stopCamera(self):
        self.timer.stop()

    def getPos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        print(x, y, self.last_frame[y, x])

        # https://stackoverflow.com/questions/41103148/capture-webcam-video-using-pyqt
    def nextFrameSlot(self):
        rval, frame = self.vc.read()
        self.last_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)
        self.label.mousePressEvent = self.getPos


def main():
    app = QApplication(sys.argv)
    ex = UI_Window()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()