from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
import qdarkstyle
import sys
import classification
import mulit_brightness
import multi_contrast
import multi_gaussian_blur
import matplotlib.pyplot as plt




class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image_labels = []
        self.image_paths = []
        # Remove the window frame
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Set the dark mode style
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.pointer = 0
        self.flag1 = False
        self.flag2 = False

        # Set window title and size
        self.setWindowTitle('Parallel Processing')
        self.setGeometry(0, 0, 1340, 800)

        # Set window icon
        self.setWindowIcon(QIcon('Images/logo.png'))

        # Create labels to display the selected images
        self.res = QLabel(self)
        self.res.setGeometry(50, 100, 400, 200)
        pixmap = QPixmap('Images/error-image.png')

        self.img = QLabel(self)
        self.img.setGeometry(70, 180, 400, 400)
        self.img.setPixmap(pixmap)
        self.img.setScaledContents(True)

        self.img_com = QLabel(self)
        self.img_com.setGeometry(550, 180, 400, 400)
        self.img_com.setPixmap(QPixmap('Images/error-image.png'))
        self.img_com.setScaledContents(True)

        # Center window on screen
        self.center()

        # create widgets
        self.create_widgets()

    def center(self):
        # Get the screen geometry
        screen = QDesktopWidget().screenGeometry()

        # Calculate the center point
        center_x = (screen.width() - self.width()) // 2
        center_y = (screen.height() - self.height()) // 2

        # Move the window to the center
        self.move(center_x, center_y)

    def create_widgets(self):
        # layout for the main window
        layout = QVBoxLayout(self)

        # Add a label
        self.label = QLabel("Image Processing Project", self)
        self.label.setStyleSheet("font-size: 20pt; font-weight: bold;")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label, alignment=Qt.AlignTop)

        self.select_file_button = QPushButton('Select Image', self)
        self.select_file_button.setStyleSheet("font-size: 12pt;")
        self.select_file_button.resize(175, 50)
        self.select_file_button.move(1050, 150)
        self.select_file_button.clicked.connect(self.select_file)

        self.classify_button = QPushButton('Classify Image', self)
        self.classify_button.setStyleSheet("font-size: 12pt;")
        self.classify_button.resize(175, 50)
        self.classify_button.move(1050, 250)
        self.classify_button.clicked.connect(self.apply_classification)

        self.combo_algorithms = QComboBox(self)
        self.combo_algorithms.resize(175, 50)
        self.combo_algorithms.move(1050, 350)
        self.combo_algorithms.addItems(['Contrast Enhance', 'Gaussian blur','Brightness'])
        default_index = self.combo_algorithms.findText('Contrast Enhance')
        if default_index != -1:
            self.combo_algorithms.setCurrentIndex(default_index)

        self.apply_Algo_button = QPushButton('Apply Algorithm ', self)
        self.apply_Algo_button.setStyleSheet("font-size: 12pt;")
        self.apply_Algo_button.resize(175, 50)
        self.apply_Algo_button.move(1050, 450)
        self.apply_Algo_button.clicked.connect(self.apply_algorithm)

        self.show_button = QPushButton('Display', self)
        self.show_button.setStyleSheet("font-size: 12pt;")
        self.show_button.resize(175, 50)
        self.show_button.move(1050, 550)
        self.show_button.clicked.connect(self.show_fig)

        self.close_button = QPushButton('Quit', self)
        self.close_button.setStyleSheet("font-size: 12pt;")
        self.close_button.resize(175, 50)
        self.close_button.move(1050, 650)
        self.close_button.clicked.connect(self.close)

        self.dark = QRadioButton("Dark mode", self)
        self.dark.move(1200, 730)
        self.dark.setChecked((True))
        self.dark.toggled.connect(self.setDark)

        self.light = QRadioButton("Light mode", self)
        self.light.move(1200, 760)
        self.light.toggled.connect(self.setLight)

    # Define a method to select an image
    def select_file(self):
        global filename

        self.filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        filename=self.filename
        # If an image is selected, display it
        if self.filename:
            pixmap = QPixmap(self.filename)
            self.img.setPixmap(pixmap)
            self.img.setScaledContents(True)

            self.flag1 = True

            errcomimg = QPixmap('Images/error-image.png')
            self.img_com.setPixmap(errcomimg)
            self.img_com.setScaledContents(True)

    def enhance_algorithms(self):
        if self.flag1:
            loading = QPixmap('Images/loading.jpg')
            self.img_com.setPixmap(loading)
            self.img_com.setScaledContents(True)
            self.img_com.show()
            self.flag2 = True

            if selected_item == 'Gaussian blur':
                # Apply blur algorithm
                print("Applying Gaussian blur Algorithm")
                gaussian_blur_algo(self.filename)

            elif selected_item == 'Contrast Enhance':
                # Apply edge enhance algorithm
                print("Applying Edge Enhance Algorithm")
                contrast_algo(self.filename)

            elif selected_item == 'Brightness':
                # Apply brightness algorithm
                print("Applying Brightness Algorithm")
                brightness_algo(self.filename)

            com = QPixmap("Results/enhanced_img.jpg")
            self.img_com.setPixmap(com)
            self.img_com.setScaledContents(True)

            self.orgi = QLabel(self)
            self.orgi.setGeometry(100, 890, 700, 50)
            self.orgi.setStyleSheet("font-size: 16pt; font-weight: bold;")
            self.orgi.show()

            self.pcom = QLabel(self)
            self.pcom.setGeometry(130, 890, 700, 50)
            self.pcom.setStyleSheet("font-size: 16pt; font-weight: bold;")
            self.pcom.show()

    def show_fig(self):
        try:
            if fig is not None:
                fig.show()
        except NameError:
            print("fig is not defined")

    def apply_algorithm(self):
        global selected_item
        selected_item = self.combo_algorithms.currentText()
        self.enhance_algorithms()

    def apply_classification(self):
        try:
            if filename is not None:
                classification(self.filename)
        except NameError:
            print("filename is not defined")



    # Define a function for setting the dark theme
    def setDark(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Define a function for setting the light theme
    def setLight(self):

        self.setStyleSheet('')


def brightness_algo(path):

    img = plt.imread(path)

    image_enhanced=mulit_brightness.main(path)
    # Plot the original and compressed images side by side
    global fig
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    print(f'{path}')
    ax[0].imshow(img)
    ax[1].imshow(image_enhanced)

    print("Done :)")

def contrast_algo(path):

    img = plt.imread(path)

    image_enhanced=multi_contrast.main(path)
    # Plot the original and compressed images side by side

    global fig
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img)

    ax[1].imshow(image_enhanced)

    print("Done :) (contrast)")

def gaussian_blur_algo(path):

    img = plt.imread(path)

    image_enhanced=multi_gaussian_blur.main(path)
    # Plot the original and compressed images side by side

    global fig
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img)

    ax[1].imshow(image_enhanced)

    print("Done :) (gaussian blur)")

def image_classification(path):
    # img = plt.imread(path)

    # Put here call of classification code ...

    classified_image = classification.main(path)
    # Plot the original and compressed images side by side

    global fig
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(classified_image)

    fig.show()

    print("Done :) (contrast)")




if __name__ == '__main__':
    # Create a QApplication instance
    app = QApplication(sys.argv)
    # Create an instance of our window
    window = MyWindow()

    # Show the window
    window.show()
    # Start the event loop and exit the application when the loop is finished
    sys.exit(app.exec_())
