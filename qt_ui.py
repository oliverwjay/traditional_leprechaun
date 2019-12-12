"""
UI Modified from https://github.com/yushulx/python/blob/master/examples/qt/barcode-reader.py
for CS549 group M
"""

import sys
from PyQt5.QtGui import QPixmap, QImage

from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit, \
    QSizePolicy, QMessageBox, QHBoxLayout, QRadioButton, QSlider
from PyQt5.QtCore import Qt, QStringListModel, QSize, QTimer

from detection_controller import DetectionController


class UI_Window(QWidget):

    def __init__(self):
        """
        Build user interface
        """
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

        # Add a button for saving model
        saveButton = QPushButton("Save Model")
        saveButton.clicked.connect(self.saveModel)
        layout.addWidget(saveButton)

        # Add a button
        button_layout = QHBoxLayout()

        btnCamera = QPushButton("Start camera")
        btnCamera.clicked.connect(self.openCamera)
        button_layout.addWidget(btnCamera)

        btnCamera = QPushButton("Stop camera")
        btnCamera.clicked.connect(self.stopCamera)
        button_layout.addWidget(btnCamera)

        layout.addLayout(button_layout)

        center_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Add a label to hold raw image
        self.raw_frame = QLabel()
        self.raw_frame.setFixedSize(640, 360)
        left_layout.addWidget(self.raw_frame)

        # Add a label to hold filtered image
        self.filtered_frame = QLabel()
        self.filtered_frame.setFixedSize(640, 360)
        left_layout.addWidget(self.filtered_frame)

        # Set last frame
        self.last_frame = None

        # Create controller
        self.det_controller = DetectionController()

        # Create radio buttons
        for i, comp in enumerate(self.det_controller.object.components.keys()):
            radiobutton = QRadioButton(comp)
            if i == 0:
                radiobutton.setChecked(False)
                self.det_controller.selected_component = comp
            radiobutton.component = comp
            radiobutton.toggled.connect(self.compChanged)
            right_layout.addWidget(radiobutton)

        # Create morphology sliders
        slider_stats = self.det_controller.get_slider_values()
        self.sliders = {}
        for slider_name in slider_stats.keys():
            slider_label = QLabel()
            slider_label.setText(slider_name)
            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.setSingleStep(2)
            slider.slider_name = slider_name
            slider.valueChanged.connect(self.sliderChanged)
            slider.setValue(slider_stats[slider_name])
            self.sliders[slider_name] = slider
            right_layout.addWidget(slider_label)
            right_layout.addWidget(slider)

        # Add store contour button
        clearBtn = QPushButton("Clear data for color")
        clearBtn.pressed.connect(self.clearColor)
        right_layout.addWidget(clearBtn)

        # Add store contour button
        sizeBtn = QPushButton("Save contour")
        sizeBtn.pressed.connect(self.saveContour)
        right_layout.addWidget(sizeBtn)

        # Add store contour button
        sizeBtn = QPushButton("Clear contours")
        sizeBtn.pressed.connect(self.clearContour)
        right_layout.addWidget(sizeBtn)

        # Add a text area
        self.results = QTextEdit()
        right_layout.addWidget(self.results)

        # Add click message field
        self.click_text = QLabel("")
        left_layout.addWidget(self.click_text)

        # Merge layouts
        center_layout.addLayout(left_layout)
        center_layout.addLayout(right_layout)
        layout.addLayout(center_layout)

        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle("Leprechaun Detector")
        self.setFixedSize(1000, 900)

    def clearColor(self):
        """
        Clear data for a color
        :return:
        """
        self.det_controller.clear_color()

    def saveContour(self):
        """
        Save a contour to model
        :return:
        """
        self.det_controller.save_sizes()

    def clearContour(self):
        """
        Clear contours for the selected component
        :return:
        """
        self.det_controller.clear_sizes()

    def sliderChanged(self, value):
        """
        Handle value change in a slider
        :param value: New value
        :return:
        """
        self.det_controller.set_slider(self.sender().slider_name, value)

    def compChanged(self):
        """
        Handle a change to the selected component
        :return:
        """
        radiobutton = self.sender()
        if radiobutton.isChecked():
            print(f"Changed to {radiobutton.component}")
            self.det_controller.selected_component = radiobutton.component
            slider_stats = self.det_controller.get_slider_values()
            for slider_name in slider_stats.keys():
                slider = self.sliders[slider_name]
                slider.setFocusPolicy(Qt.StrongFocus)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(10)
                slider.setSingleStep(2)
                slider.slider_name = slider_name
                slider.setValue(slider_stats[slider_name])

    def closeEvent(self, event):
        """
        Handle closing the app
        :param event:
        :return:
        """
        msg = "Close the app?"
        reply = QMessageBox.question(self, 'Message',
                                     msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            self.stopCamera()
        else:
            event.ignore()

    def resizeImage(self, filename):
        """
        Resizes the image to display
        :param filename:
        :return:
        """
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
        """
        Handles selecting an image to process
        :return:
        """
        self.stopCamera()
        # Load an image file.
        filename = QFileDialog.getOpenFileName(self, 'Open file',
                                               'E:\\Program Files (x86)\\Dynamsoft\\Barcode Reader 7.1\\Images',
                                               "Barcode images (*)")
        # Update
        raw, filtered = self.det_controller.process_from_file(filename[0])
        self.updateFrameDisplay(raw, filtered)
        self.timer.start(1000. / 24)

    def openCamera(self):
        """
        Handle starting the camera
        :return:
        """
        self.det_controller.set_input_to_camera()
        self.timer.start(1000. / 24)

    def stopCamera(self):
        """
        Handle pausing the camera
        :return:
        """
        self.det_controller.set_input_to_static()

    def saveModel(self):
        """
        Handle saving the model
        :return:
        """
        self.det_controller.object.save()

    def getImgPos(self, event):
        """
        Handle clicking the image
        :param event:
        :return:
        """
        x = event.pos().x()
        y = event.pos().y()
        result = self.det_controller.handle_click(x, y)
        self.click_text.setText(result)

    def getContourPos(self, event):
        """
        Handle clicking a contour
        :param event:
        :return:
        """
        x = event.pos().x()
        y = event.pos().y()
        self.det_controller.save_contour(x, y)

        # https://stackoverflow.com/questions/41103148/capture-webcam-video-using-pyqt
    def nextFrameSlot(self):
        """
        Process next frame
        :return:
        """
        # Get frames
        raw, filtered = self.det_controller.update_image()
        self.updateFrameDisplay(raw, filtered)

    def updateFrameDisplay(self, raw, filtered):
        """
        Update displayed frame
        :param raw: Raw frame to display
        :param filtered: Processed frame to display
        :return:
        """

        # Convert to QImage
        raw_image = QImage(raw, raw.shape[1], raw.shape[0], QImage.Format_RGB888)
        filtered_image = QImage(filtered, filtered.shape[1], filtered.shape[0], QImage.Format_RGB888)

        # Add to frame as pixmap
        raw_pixmap = QPixmap.fromImage(raw_image)
        filtered_pixmap = QPixmap.fromImage(filtered_image)
        self.raw_frame.setPixmap(raw_pixmap)
        self.filtered_frame.setPixmap(filtered_pixmap)

        # Add click handlers
        self.raw_frame.mousePressEvent = self.getImgPos
        self.filtered_frame.mousePressEvent = self.getContourPos


def main():
    """
    Start applicaiton
    :return:
    """
    app = QApplication(sys.argv)
    ex = UI_Window()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()