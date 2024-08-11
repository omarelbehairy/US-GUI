import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QScrollArea
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class ImageVideoApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image and Video Display Example")
        self.setGeometry(100, 100, 1200, 600)

        # Create a horizontal layout to place image and video side by side
        hbox_main = QHBoxLayout()

        # Create a vertical layout for the image section
        vbox_image = QVBoxLayout()

        # Create a scroll area for the image
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        # Create a label to display the image
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.label)
        
        vbox_image.addWidget(self.scroll_area)

        # Create a horizontal layout for image buttons
        hbox_image_buttons = QHBoxLayout()

        # Create an upload image button
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)
        hbox_image_buttons.addWidget(self.upload_button)

        # Create a zoom in button
        self.zoom_in_button = QPushButton("Zoom In", self)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        hbox_image_buttons.addWidget(self.zoom_in_button)

        # Create a zoom out button
        self.zoom_out_button = QPushButton("Zoom Out", self)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        hbox_image_buttons.addWidget(self.zoom_out_button)

        # Add the image buttons to the vertical layout
        vbox_image.addLayout(hbox_image_buttons)

        # Create a generate ultrasound image button
        self.ultrasound_button = QPushButton("Generate Ultrasound Image", self)
        self.ultrasound_button.clicked.connect(self.generate_ultrasound_image)
        vbox_image.addWidget(self.ultrasound_button)

        # Add the image section to the main horizontal layout
        hbox_main.addLayout(vbox_image)

        # Create a vertical layout for the video section
        vbox_video = QVBoxLayout()

        # Create a video widget
        self.video_widget = QVideoWidget(self)

        # Create a media player
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        vbox_video.addWidget(self.video_widget)

        # Create an upload video button
        self.upload_video_button = QPushButton("Upload Video", self)
        self.upload_video_button.clicked.connect(self.upload_video)
        vbox_video.addWidget(self.upload_video_button)

        # Add the video section to the main horizontal layout
        hbox_main.addLayout(vbox_video)

        # Set the layout for the main window
        self.setLayout(hbox_main)

        # Initialize the current pixmap
        self.pixmap = None
        self.current_scale = 1.0

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        
        if file_name:
            self.pixmap = QPixmap(file_name)
            self.current_scale = 1.0
            self.update_image_display()

    def zoom_in(self):
        if self.pixmap:
            self.current_scale += 0.1
            self.update_image_display()

    def zoom_out(self):
        if self.pixmap:
            self.current_scale -= 0.1
            if self.current_scale < 0.1:
                self.current_scale = 0.1
            self.update_image_display()

    def update_image_display(self):
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(self.pixmap.size() * self.current_scale, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)
            self.label.resize(scaled_pixmap.size())

    def upload_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)", options=options)

        if file_name:
            self.media_player.setMedia(QMediaContent(file_name))
            self.media_player.play()

    def generate_ultrasound_image(self):
        # Parameters
        num_lines = 128       # Number of scan lines
        num_samples = 1024    # Number of samples per scan line
        organ_radius = 0.5    # Radius of the circular organ
        sampling_frequency = 40e6  # 40 MHz

        echo_data = generate_organ_echoes(num_lines, num_samples, organ_radius)
        envelope_data = np.abs(hilbert(echo_data, axis=1))
        log_compressed_data = 20 * np.log10(envelope_data + 1e-6)
        log_compressed_data -= log_compressed_data.min()
        log_compressed_data /= log_compressed_data.max()

        # Convert the generated image to a QImage
        height, width = log_compressed_data.shape
        img = (log_compressed_data * 255).astype(np.uint8)
        qimg = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(qimg)

        self.current_scale = 1.0
        self.update_image_display()

# Simulate an organ as a circle with different echo intensities inside and outside
def generate_organ_echoes(num_lines, num_samples, organ_radius, noise_level=0.1):
    x = np.linspace(-1, 1, num_lines)
    y = np.linspace(-1, 1, num_samples)
    xx, yy = np.meshgrid(x, y)
    organ = np.sqrt(xx**2 + yy**2) <= organ_radius
    echo_data = np.zeros_like(organ, dtype=np.float32)
    echo_data[organ] = 1.0
    echo_data += noise_level * np.random.randn(*echo_data.shape)
    echo_data = gaussian_filter(echo_data, sigma=3)
    return echo_data.T

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageVideoApp()
    ex.show()
    sys.exit(app.exec_())
