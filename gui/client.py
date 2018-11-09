import os
import sys
import socket
import queue
import threading
import time
import atexit
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class DataGen(object):
    """Data Generator/Extractor class"""

    def __init__(self):
        file_path = self.get_file_path()

        try:
            self.dataset = pd.read_csv(file_path)

        except Exception:
            sys.exit(1)

        self.neurons = self.prompt_data_selection("Neurons", "Choose the neurons to plot")
        self.behaviors = self.prompt_data_selection("Behaviors", "Choose your behaviors")
        self.parse_data_selection()
        print(self.neurons)
        print(self.behaviors)
        print(type(self.behaviors))

        self.dataset.fillna(0)
        self.neuron_col_vectors = self.dataset[self.neurons]

        self.global_ymax = self.neuron_col_vectors.max().max()
        self.all_behavior_intervals = self.get_behavior(self.dataset)
        del self.dataset

    def get_file_path(self):
        file_path = ""

        while not file_path:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            print(file_path)
            root.destroy()

        return file_path

    def prompt_data_selection(self, msg1, msg2):
        root = tk.Tk()
        user_input = ""
        while not user_input:
            user_input = simpledialog.askstring(msg1, msg2, parent=root)
            print(user_input)

        root.destroy()
        return user_input

    def parse_data_selection(self):
        self.neurons = self.neurons.replace(' ', '')
        self.neurons = self.neurons.split(',')

        self.behaviors = self.behaviors.replace(' ', '')
        self.behaviors = self.behaviors.split(',')

    def get_behavior(self, dataset):
        all_behavior_intervals = []

        for behavior in self.behaviors:
            curr_beh_epochs = self.extract_epochs(dataset, behavior)
            curr_beh_intervals = self.filter_epochs(curr_beh_epochs[1], framerate=1, seconds=1)
            all_behavior_intervals.append(curr_beh_intervals)

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

class Client(object):

    def __init__(self, address, port, q):
        self.q = q

        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        server_address = (address, port)
        print("Connecting to: {}".format(server_address))
        self.sock.connect(server_address)

        t = threading.Thread(target=self.data_receiver, args=())
        t.daemon = True
        t.start()

    def data_receiver(self):
        print("new thread started")
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
        finally:
            print("closing socket")
            self.sock.close()
        
    def close_socket(self):
        self.sock.close()

from pyqt_wrapper import MyWidget

if __name__ == "__main__":
    datagen = DataGen()
    plots = datagen.get_neuron_plots()
    plot_names = datagen.neurons

    q = queue.Queue()
    client = Client("localhost", 10000, q)

    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=False) # True seems to work as well

    win = MyWidget(q, plots, plot_names)
    win.show()
    win.resize(800,600)
    win.raise_()
    app.exec_()
    sys.exit(app.exec_())
