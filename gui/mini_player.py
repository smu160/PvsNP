#
# PvsNP: toolbox for reproducible analysis & visualization of neurophysiological data.
# Copyright (C) 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""
This module contains a bare-bones vlc player class to stream videos.
"""

__author__ = "Saveliy Yusufov"
__date__ = "25 December 2018"
__license__ = "GPL"
__maintainer__ = "Saveliy Yusufov"
__email__ = "sy2685@columbia.edu"

import os
import sys
import queue
import platform

from PyQt5 import QtWidgets, QtGui, QtCore
import vlc
from network import Client


class Communicate(QtCore.QObject):

    update_mp = QtCore.pyqtSignal(str)


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
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.get)
        self.timer.timeout.connect(self.update_statusbar)

        self.jump_table = {'P': self.mediaplayer.play,
                           'p': self.mediaplayer.pause,
                           'S': self.mediaplayer.stop,
                           '<': self.slow_down,
                           '>': self.speed_up
                          }

        self.c = Communicate()
        self.c.update_mp[str].connect(self.update_ui)
        self.data_queue = data_queue
        self.timer.start()

    def init_ui(self):
        """Set up the user interface
        """
        self.widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.widget)

        # In this widget, the video will be drawn
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

    def get(self):
        try:
            val = self.data_queue.get(block=False)
            self.c.update_mp.emit(val)
        except queue.Empty:
            return

    def update_ui(self, val):
        """Updates the UI on every time interval"""
        try:
            self.jump_table[val]()
        except KeyError:
            val = int(val)
            if val != self.mediaplayer.get_time():
                self.mediaplayer.set_time(val)

    def update_statusbar(self):
        """Updates the statusbar with the current media's time"""
        mtime = QtCore.QTime(0, 0, 0, 0)
        time = mtime.addMSecs(self.mediaplayer.get_time())
        self.statusbar.showMessage(time.toString())

    def slow_down(self):
        self.mediaplayer.set_rate(self.mediaplayer.get_rate() * 0.5)

    def speed_up(self):
        self.mediaplayer.set_rate(self.mediaplayer.get_rate() * 2)

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
