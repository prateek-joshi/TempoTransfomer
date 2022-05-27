from app import Window
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
import sys


if __name__=='__main__':
    App = QApplication(sys.argv)
    App.setApplicationName('Audio Player')
    App.setStyle('Fusion')
    Equalizer = Window()
    Equalizer.resize(1080, 480)
    Equalizer.show()

    sys.exit(App.exec_())