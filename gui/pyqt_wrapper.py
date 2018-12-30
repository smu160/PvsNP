"""
This module contains the classes to create a plot(s) window.

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import queue
import sys
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

class MainWindow(QtWidgets.QMainWindow):
    """This class wraps the PlotWindow class to add extra functionality."""

    def __init__(self, data_q, plots, plot_names, beh_intervals=None, parent=None):
        super().__init__(parent)

        # Create plot window widget & set it as central the central widget
        self.plot_window = PlotWindow(data_q, plots, plot_names, beh_intervals=beh_intervals, parent=self)
        self.setCentralWidget(self.plot_window)

        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # Create submenu to start new processes from file menu
        adjust_plots = QtWidgets.QMenu("Adjust plots", self)
        file_menu.addMenu(adjust_plots)
        set_x_axis = QtWidgets.QAction("set x axis", self)
        set_y_axis = QtWidgets.QAction("set y axis", self)
        adjust_plots.addAction(set_x_axis)
        adjust_plots.addAction(set_y_axis)
        set_x_axis.triggered.connect(self.on_set_x_axis)
        set_y_axis.triggered.connect(self.on_set_y_axis)

        # Create and connect action to close the plot window
        close_action = QtWidgets.QAction("Close", self)
        file_menu.addAction(close_action)
        close_action.triggered.connect(sys.exit)

    def on_set_x_axis(self):
        lower_bound, ok = QtWidgets.QInputDialog.getInt(self, "X-axis", "lower bound:")

        if ok:
            upper_bound, ok_2 = QtWidgets.QInputDialog.getInt(self, "X-axis", "upper bound:")
            if ok_2:
                if lower_bound <= upper_bound:
                    for plot_item in self.plot_window.plot_items:
                        plot_item.setRange(xRange=[lower_bound, upper_bound], padding=0)

                    print("Changed x axis", file=sys.stderr)
            else:
                return
        else:
            return

    def on_set_y_axis(self):
        print("Set y-axis... yay!")

class PlotWindow(pg.GraphicsWindow):
    """This class holds plots and should be nested in a MainWindow."""

    def __init__(self, data_q, plots, plot_names, beh_intervals=None, parent=None):
        super().__init__(parent=parent)

        self.graphics_layout = pg.GraphicsLayout(border=(0, 0, 0))
        self.graphics_layout.layout.setSpacing(0)
        self.graphics_layout.layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralItem(self.graphics_layout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.start()
        self.timer.timeout.connect(self.on_new_data)

        self.plot_items = []
        self.vertical_lines = []
        self.plots = plots
        self.add_plots(plot_names, beh_intervals)
        self.data_q = data_q

    def on_new_data(self):
        """Called on timer interval to get data from queue & update all plots.
        """
        try:
            val = self.data_q.get(block=False)
        except queue.Empty:
            return

        for v_line in self.vertical_lines:
            v_line.setValue(val)
            QtCore.QCoreApplication.processEvents()

    def add_plots(self, plot_names, all_beh_intervals):
        """Creates and set all the plots"""

        # Find the maximum value of the x-axis for all plots
        x_max = len(self.plots[0])

        # Pen to be used for vertical lines
        pen = pg.mkPen('r', width=2)

        for i, plot in enumerate(self.plots):
            plot_item = self.graphics_layout.addPlot(title="plot {}".format(plot_names[i]), row=i, col=0)
            plot_item.plot(plot, pen=pg.mkPen('b', width=2))

            # Set the domain and range for each plot
            y_max = plot.max()
            plot_item.setRange(xRange=[0, x_max], yRange=[0, y_max], padding=0)

            # Add background color(s) (color coded by behavior) to each plot
            if all_beh_intervals:
                for behavior_intervals, color in all_beh_intervals:
                    for interval in behavior_intervals:
                        rgn = pg.LinearRegionItem(values=[interval[0], interval[-1]], movable=False)
                        rgn.lines[0].setPen((255, 255, 255, 5))
                        rgn.lines[1].setPen((255, 255, 255, 5))
                        rgn.setBrush(pg.mkBrush(color))
                        plot_item.addItem(rgn)

            # Create and add vertical line that scrolls
            self.vertical_line = plot_item.addLine(x=0, pen=pen)
            self.vertical_lines.append(self.vertical_line)

            # Store references to each PlotItem
            self.plot_items.append(plot_item)


if __name__ == "__main__":
    pg.setConfigOption("background", 'w')
    pg.setConfigOption("foreground", 'k')
