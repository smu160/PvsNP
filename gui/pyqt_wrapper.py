"""
This module contains the classes to create a plot(s) window.

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import queue
import sys
from PyQt5 import QtCore, QtWidgets
from data_dialogs import AxisDialog
import pyqtgraph as pg

class MainWindow(QtWidgets.QMainWindow):
    """This class wraps the PlotWindow class to add extra functionality."""

    def __init__(self, data_queue, plots, plot_names, beh_intervals=None, parent=None):
        super().__init__(parent)

        # Create plot window widget & set it as the central widget
        self.plot_window = PlotWindow(data_queue, plots, plot_names, beh_intervals=beh_intervals, parent=self)
        self.setCentralWidget(self.plot_window)

        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready")

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
        set_x_axis.triggered.connect(self.on_set_axis)
        set_y_axis.triggered.connect(self.on_set_y_axis)

        # Create and connect action to close the plot window
        close_action = QtWidgets.QAction("Close", self)
        file_menu.addAction(close_action)
        close_action.triggered.connect(sys.exit)

    def on_set_axis(self, axis=0):
        """Handles both actions in the 'Adjust plots' submenu"""

        lower_bound, upper_bound = show_axis_dialog()

        if lower_bound and upper_bound:

            # lineedit returns str, so first convert both to int
            lower_bound = int(lower_bound)
            upper_bound = int(upper_bound)

            for plot_item in self.plot_window.plot_items:
                if axis:
                    plot_item.setRange(yRange=[lower_bound, upper_bound], padding=0)
                else:
                    plot_item.setRange(xRange=[lower_bound, upper_bound], padding=0)

    def on_set_y_axis(self):
        self.on_set_axis(axis=1)


def show_axis_dialog():
    """Display the behavior dialog to the user & let user choose colors.
    """
    # app = QtWidgets.QApplication(sys.argv)
    axis_dialog = AxisDialog()
    axis_dialog.exec_()
    axis_dialog.show()
    return axis_dialog.lower_bound, axis_dialog.upper_bound


class PlotWindow(pg.GraphicsWindow):
    """This class holds plots and should be nested in a MainWindow."""

    changed_behavior = QtCore.pyqtSignal('QString')

    def __init__(self, data_queue, plots, plot_names, beh_intervals=None, parent=None):
        super().__init__(parent=parent)
        self.parent = parent

        self.graphics_layout = pg.GraphicsLayout(border=(0, 0, 0))
        self.graphics_layout.layout.setSpacing(0)
        self.graphics_layout.layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralItem(self.graphics_layout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.on_new_data)

        self.behavior_time = {}
        self.plot_items = []
        self.vertical_lines = []
        self.plots = plots
        self.add_plots(plot_names, beh_intervals)

        # Convert all sets to strings
        if beh_intervals:
            self.behavior_time = {time: ', '.join(behavior) for time, behavior in self.behavior_time.items()}
        else:
            self.behavior_time = None

        self.data_queue = data_queue
        self.timer.start()

    def on_new_data(self):
        """Called on timer interval to get data from queue & update all plots.
        """
        try:
            val = self.data_queue.get(block=False)
        except queue.Empty:
            QtCore.QCoreApplication.processEvents()
            return

        if val in ('<', '>', 'P', 'p'):
            return
        if val == 'S':
            for v_line in self.vertical_lines:
                v_line.setValue(0)
            if self.behavior_time:
                self.parent.statusbar.showMessage(self.behavior_time[0])
            return

        val = int(val)
        val //= 100
        for v_line in self.vertical_lines:
            v_line.setValue(val)
        if self.behavior_time:
            self.parent.statusbar.showMessage(self.behavior_time[val])

        QtCore.QCoreApplication.processEvents()

    def add_plots(self, plot_names, all_beh_intervals):
        """Creates and set all the plots"""

        # Find the maximum value of the x-axis for all plots
        x_max = len(self.plots[0])

        # Pen to be used for vertical lines
        pen = pg.mkPen('r', width=2)

        # Get background color(s) (color coded by behavior) to add to each plot
        list_of_lists_of_rgns = [[] for _ in self.plots]
        if all_beh_intervals:
            for behavior_intervals, behavior, color in all_beh_intervals:
                for interval in behavior_intervals:

                    # Add the interval times and the corresponding behavior
                    # to track it for the status bar
                    for time in interval:
                        if time not in self.behavior_time:
                            self.behavior_time[time] = {behavior}
                        else:
                            self.behavior_time[time].add(behavior)

                    # Add rgn object for each plot
                    # (Since copies are not possible)
                    for inner_lst in list_of_lists_of_rgns:
                        rgn = pg.LinearRegionItem(values=[interval[0], interval[-1]], movable=False)
                        rgn.lines[0].setPen((255, 255, 255, 5))
                        rgn.lines[1].setPen((255, 255, 255, 5))
                        rgn.setBrush(pg.mkBrush(color))
                        inner_lst.append(rgn)

        for i, plot in enumerate(self.plots):
            plot_item = self.graphics_layout.addPlot(title="{}".format(plot_names[i]), row=i, col=0)

            # Causes auto-scale button (‘A’ in lower-left corner)
            # to be hidden for this PlotItem
            plot_item.hideButtons()

            plot_item.setMouseEnabled(x=True, y=False)

            for rgn in list_of_lists_of_rgns[i]:
                plot_item.addItem(rgn)

            plot_item.plot(plot, pen=pg.mkPen((0, 0, 0), width=2))

            # Get the max value in the time series to set the yRange, below
            y_max = plot.max()

            # Set the domain and range for each plot
            plot_item.setRange(xRange=[0, x_max], yRange=[-1, y_max], padding=0)

            # Create and add vertical line that scrolls
            self.vertical_line = plot_item.addLine(x=0, pen=pen)
            self.vertical_lines.append(self.vertical_line)

            # Store references to each PlotItem
            self.plot_items.append(plot_item)


if __name__ == "__main__":
    pg.setConfigOption("background", 'w')
    pg.setConfigOption("foreground", 'k')
