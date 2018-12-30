"""
This module contains the classes for receiving, extracting, and plotting data.

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import sys
import platform
import socket
import queue
import threading

from tkinter import filedialog
import tkinter as tk
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui
import pyqtgraph as pg
from pyqt_wrapper import MainWindow
from data_dialogs import DataDialog, ColorsDialog

class DataGen:
    """Data Generator/Extractor class"""

    def __init__(self):
        file_path = self.get_file_path()

        try:
            self.dataset = pd.read_csv(file_path)
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
            # ammend each color with a transparency value.
            if self.choose_colors:
                self.behaviors = self.show_behavior_colors_dialog()
                for behavior, color in self.behaviors.items():
                    temp = list(color)
                    temp[3] = 40
                    print(temp)
                    self.behaviors[behavior] = temp
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
            self.behavior_intervals = self.get_behavior(self.dataset)
        else:
            self.behavior_intervals = None

        # We no longer need the dataframe
        del self.dataset

    def get_file_path(self):
        file_path = ""

        while not file_path:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            root.destroy()

        return file_path

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

    def get_behavior(self, dataset):
        all_behavior_intervals = []

        for behavior, color in self.behaviors.items():
            curr_beh_epochs = self.extract_epochs(dataset, behavior)
            curr_beh_intervals = self.filter_epochs(curr_beh_epochs[1], framerate=1, seconds=1)
            all_behavior_intervals.append((curr_beh_intervals, color))

        return all_behavior_intervals

    def extract_epochs(self, dataset, behavior):
        dataframe = dataset
        dataframe["block"] = (dataframe[behavior].shift(1) != dataframe[behavior]).astype(int).cumsum()
        df = dataframe.reset_index().groupby([behavior, "block"])["index"].apply(np.array)
        return df

    def filter_epochs(self, interval_series, framerate=10, seconds=1):
        intervals = []

        for interval in interval_series:
            if len(interval) >= framerate*seconds:
                intervals.append(interval)

        return intervals

    def get_neuron_plots(self):
        plots = []

        for col in self.neuron_col_vectors:
            plots.append(self.neuron_col_vectors[col].values)

        return plots

class Client:

    def __init__(self, address, port, q):
        self.q = q

        if platform.system() == "Windows":

            # Create a TCP/IP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect the socket to the port where the server is listening
            server_address = (address, port)
            print("Connecting to {} port {}".format(server_address[0], server_address[1]))
            self.sock.connect(server_address)
        else:

            # Create a UDS socket
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            # Connect the socket to the port where the server is listening
            server_address = "./uds_socket"
            print("New client connecting to {}".format(server_address))

            try:
                self.sock.connect(server_address)
            except socket.error as msg:
                print(msg, file=sys.stderr)
                sys.exit(1)

        t = threading.Thread(target=self.data_receiver, args=())
        t.daemon = True
        t.start()

    def data_receiver(self):
        print("New data receiver thread started...")
        try:
            while True:
                data = self.sock.recv(4)
                if data:
                    data = data.decode()

                data = data.split(',')
                if data:
                    for num in data:
                        if num:
                            if num == 'd':
                                self.q.queue.clear()
                            else:
                                self.q.put(int(num))
        except:
            print("Closing socket...", file=sys.stderr)
            self.sock.close()
            return

def main():
    datagen = DataGen()
    plots = datagen.get_neuron_plots()
    plot_names = datagen.neurons

    q = queue.Queue()
    _ = Client("localhost", 10000, q)

    # Create new plot window
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True) # set to True for higher quality plots
    main_window = MainWindow(q, plots, plot_names, beh_intervals=datagen.behavior_intervals)
    main_window.show()
    main_window.resize(800, 600)
    main_window.raise_()
    sys.exit(app.exec_())

if __name__ == "__main__":
    pg.setConfigOption("background", 'w')
    pg.setConfigOption("foreground", 'k')
    main()
