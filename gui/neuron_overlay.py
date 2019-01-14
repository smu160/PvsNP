# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:32:12 2019

@Author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import random

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush, QPainter, QPalette


class Overlay(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        palette = QPalette(self.palette())
        palette.setColor(palette.Background, Qt.transparent)
        self.setPalette(palette)

        # Set (x, y) as its top-left corner and the given width and height
        parent_geometry = self.parent().geometry()
        top_left_x = parent_geometry.topLeft().x()
        top_left_y = parent_geometry.topLeft().y()
        width = parent_geometry.width()
        height = parent_geometry.height()
        self.setGeometry(top_left_x, top_left_y, width, height)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QBrush(QColor(255, 255, 255, 10)))
        painter.setPen(Qt.red)

        size = self.parent().size()
        # print(size)
        for _ in range(5000):
            x = random.randint(1, size.width()-1)
            y = random.randint(1, size.height()-1)
            painter.drawPoint(x, y)

        painter.end()

    def showEvent(self, event):
        self.timer = self.startTimer(50)
        self.counter = 0

    def timerEvent(self, event):
        self.counter += 1
        self.update()
