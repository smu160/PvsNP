import os
import pprint
import random
import sys
import wx
import wx.media
import pandas as pd

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas, NavigationToolbar2WxAgg as NavigationToolbar
import numpy as np
import pylab

from video_player import VideoPanel
from dynamic_graph import GraphPanel

class GraphFrame(wx.Frame):
    """The main frame of the application"""

    title = "Calcium Imaging Visualizer"

    def __init__(self):
        wx.Frame.__init__(self, None, -1, self.title, size=(1280, 800))

        splitter = wx.SplitterWindow(self, -1)
        self.graph_panel = GraphPanel(parent=splitter, coupled=True)
        self.video_panel = VideoPanel(parent=splitter, coupled_graph=self.graph_panel)
        splitter.SplitVertically(self.video_panel, self.graph_panel, sashPosition=0)

        self.Show()


if __name__ == "__main__":
    app = wx.App()
    app.frame = GraphFrame()
    app.MainLoop()
