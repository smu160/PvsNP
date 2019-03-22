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
This module contains the classes for receiving, extracting, and plotting data.
"""

__author__ = "Saveliy Yusufov"
__date__ = "25 December 2018"
__license__ = "GPL"
__maintainer__ = "Saveliy Yusufov"
__email__ = "sy2685@columbia.edu"

import os
import sys
import queue

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
import pyqtgraph as pg
from network import Client
from pyqt_wrapper import MainWindow
from data_dialogs import DataDialog, ColorsDialog

class DataGen:
    """Data Generator/Extractor class"""

    def __init__(self):
        self.filename = None
        self.show_file_dialog()
        if not self.filename[0]:
            sys.exit(1)

        try:
            self.dataset = pd.read_csv(self.filename[0])
        except (UnicodeDecodeError, pd.errors.ParserError) as exception:
            print(exception, file=sys.stderr)
            app = QtWidgets.QApplication([])
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText(str(exception) + "\nMake sure you choose a csv file! Try again.")
            msg.setWindowTitle("Error")
            msg.show()
            app.exec_()
            sys.exit(1)

        # Prompt user to select neurons. If the user cancels the dialog, or the
        # the user closes the dialog without selecting anything, exit.
        self.neurons = self.show_data_dialog()
        if not self.neurons:
            sys.exit(1)

        # Prompt user to select behaviors. If the user cancels or closes the
        # dialogs, then it's assumed no behaviors were chosen/needed.
        self.behaviors = self.show_data_dialog(checkbox=True)

        # If behaviors of interest were selected, then create a dictionary with
        # each behavior coupled with a corresponding color.
        if self.behaviors:

            # If user checked box to choose custom colors, display the dialog
            # for the user to choose custom colors for each behavior. Finally,
            # ammend each color with a transparency value. Note that if the user
            # doesn't choose custom colors, a random color will be assigned to
            # each behavior.
            if self.choose_colors:
                self.behaviors = self.show_behavior_colors_dialog()
                for behavior, color in self.behaviors.items():
                    self.behaviors[behavior] = color
            else:
                temp = {}
                for behavior in self.behaviors:
                    rgb = list(np.random.randint(256, size=3))
                    rgb.append(40)
                    rgba = tuple(rgb)
                    temp[behavior] = rgba

                self.behaviors = temp

        self.dataset.fillna(0)
        self.neuron_col_vectors = self.dataset[self.neurons]

        # Make sure the user actually chose colors
        if self.behaviors:
            self.behavior_intervals = self.get_behavior()
        else:
            self.behavior_intervals = None

        # We no longer need the dataframe
        del self.dataset

    def show_file_dialog(self):
        """Display a file dialog to the user & let user choose a data file.
        """
        _ = QtWidgets.QApplication(sys.argv)
        file_dialog = QtWidgets.QFileDialog()
        self.filename = file_dialog.getOpenFileName(None, "Choose a Data File", os.path.expanduser('~'))

    def show_data_dialog(self, checkbox=False):
        """Display a dialog to the user & let user choose neurons or behaviors.
        """
        app = QtWidgets.QApplication(sys.argv)
        data_dialog = DataDialog(list(self.dataset.columns), checkbox=checkbox)
        data_dialog.show()
        app.exec_()

        # If checkbox was added to dialog for choosing behaviors, then show
        # dialog for user to choose color for each behavior.
        if checkbox:
            self.choose_colors = data_dialog.choose_colors

        return data_dialog.selected_items

    def show_behavior_colors_dialog(self):
        """Display the behavior dialog to the user & let user choose colors.
        """
        app = QtWidgets.QApplication(sys.argv)
        colors_dialog = ColorsDialog(self.behaviors)
        colors_dialog.show()
        app.exec_()
        return colors_dialog.behavior_colors

    def get_behavior(self):
        """Get a list of all behavior intervals and their respective colors"""
        all_behavior_intervals = []

        for behavior, color in self.behaviors.items():
            curr_beh_epochs = self.extract_epochs(behavior)
            curr_beh_intervals = self.filter_epochs(curr_beh_epochs[1], framerate=1, seconds=1)
            all_behavior_intervals.append((curr_beh_intervals, behavior, color))

        return all_behavior_intervals

    def extract_epochs(self, behavior):
        """Extract continuous segments of the provided behavior"""
        dataframe = self.dataset
        dataframe["block"] = (dataframe[behavior].shift(1) != dataframe[behavior]).astype(int).cumsum()
        epoch_df = dataframe.reset_index().groupby([behavior, "block"])["index"].apply(np.array)
        return epoch_df

    @staticmethod
    def filter_epochs(interval_series, framerate=10, seconds=1):
        """Filter the continuous segments of behavior for a specific length"""
        return [interval for interval in interval_series if len(interval) >= framerate*seconds]

    def get_neuron_plots(self):
        """Get a list of the time series for the user-provided neurons"""
        return [self.neuron_col_vectors[col].values for col in self.neuron_col_vectors]


def main():
    """Entry point to a new window of plots"""
    datagen = DataGen()
    plots = datagen.get_neuron_plots()
    plot_names = datagen.neurons

    data_queue = queue.Queue()

    # Create new plot window
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)  # set to True for higher quality plots
    main_window = MainWindow(data_queue, plots, plot_names, beh_intervals=datagen.behavior_intervals)
    main_window.show()
    main_window.resize(800, 600)
    main_window.raise_()
    _ = Client("localhost", 10000, data_queue)
    sys.exit(app.exec_())


if __name__ == "__main__":
    pg.setConfigOption("background", 'k')
    pg.setConfigOption("foreground", 'w')
    main()
