"""This module contains a bare-bones vlc player class to stream videos.
"""

import os
import sys
import queue
import platform

from PyQt5 import QtWidgets, QtGui, QtCore
import vlc
from client import Client


class MiniPlayer(QtWidgets.QMainWindow):
    """Stripped-down PyQt5-based media player class to sync with another video.
    """

    def __init__(self, data_queue, master=None):
        QtWidgets.QMainWindow.__init__(self, master)
        self.setWindowTitle("Mini Player")
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready")

        # Create a basic vlc instance
        self.instance = vlc.Instance()

        self.media = None

        # Create an empty vlc media player
        self.mediaplayer = self.instance.media_player_new()

        self.init_ui()
        self.open_file()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_ui)

        self.data_queue = data_queue
        self.timer.start()

    def init_ui(self):
        """Set up the user interface
        """
        self.widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.widget)

        # In this widget, the video will be drawn
        if platform.system() == "Darwin":  # for MacOS
            self.videoframe = QtWidgets.QMacCocoaViewContainer(0)
        else:
            self.videoframe = QtWidgets.QFrame()

        self.palette = self.videoframe.palette()
        self.palette.setColor(QtGui.QPalette.Window, QtGui.QColor(0, 0, 0))
        self.videoframe.setPalette(self.palette)
        self.videoframe.setAutoFillBackground(True)

        self.vboxlayout = QtWidgets.QVBoxLayout()
        self.vboxlayout.addWidget(self.videoframe)
        self.widget.setLayout(self.vboxlayout)

    def open_file(self):
        """Open a media file in a MediaPlayer
        """
        dialog_txt = "Choose Media File"
        filename = QtWidgets.QFileDialog.getOpenFileName(self, dialog_txt, os.path.expanduser('~'))
        if not filename[0]:
            return

        # getOpenFileName returns a tuple, so use only the actual file name
        self.media = self.instance.media_new(filename[0])

        # Put the media in the media player
        self.mediaplayer.set_media(self.media)

        # Parse the metadata of the file
        self.media.parse()

        # Set the title of the track as the window title
        self.setWindowTitle("{}".format(self.media.get_meta(0)))

        # The media player has to be 'connected' to the QFrame (otherwise the
        # video would be displayed in it's own window). This is platform
        # specific, so we must give the ID of the QFrame (or similar object) to
        # vlc. Different platforms have different functions for this
        if platform.system() == "Linux":  # for Linux using the X Server
            self.mediaplayer.set_xwindow(int(self.videoframe.winId()))
        elif platform.system() == "Windows":  # for Windows
            self.mediaplayer.set_hwnd(int(self.videoframe.winId()))
        elif platform.system() == "Darwin":  # for MacOS
            self.mediaplayer.set_nsobject(int(self.videoframe.winId()))

        # Start playing the video as soon as it loads
        self.mediaplayer.play()

    def update_ui(self):
        self.update_statusbar()

        try:
            val = self.data_queue.get(block=False)
        except queue.Empty:
            return

        if val == '<':
            self.mediaplayer.set_rate(self.mediaplayer.get_rate() * 0.5)
            return
        if val == '>':
            self.mediaplayer.set_rate(self.mediaplayer.get_rate() * 2)
            return
        if val == 'P':
            self.mediaplayer.play()
            return
        elif val == 'p':
            self.mediaplayer.pause()
            return
        elif val == 'S':
            self.mediaplayer.stop()
            return
        else:
            val = int(val)
            if val != self.mediaplayer.get_time():
                self.mediaplayer.set_time(val)

    def update_statusbar(self):
        mtime = QtCore.QTime(0, 0, 0, 0)
        time = mtime.addMSecs(self.mediaplayer.get_time())
        self.statusbar.showMessage(time.toString())


def main():
    """Entry point for our simple vlc player
    """
    app = QtWidgets.QApplication(sys.argv)

    data_queue = queue.Queue()

    player = MiniPlayer(data_queue)
    player.show()
    player.resize(480, 480)

    _ = Client("localhost", 10000, data_queue)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
