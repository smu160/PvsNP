import wx
import wx.media
import os

import matplotlib       # Provides the graph figures
matplotlib.use('WXAgg') # matplotlib needs a GUI (layout), we use wxPython
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas

class StaticText(wx.StaticText):
    """
    A StaticText that only updates the label if it has changed, to
    help reduce potential flicker since these controls would be
    updated very frequently otherwise.
    """

    def SetLabel(self, label):
        if label == self.GetLabel():
            wx.StaticText.SetLabel(self, label)

class VideoPanel(wx.Panel):

    def __init__(self, parent, coupled_graph=None):
        wx.Panel.__init__(self, parent=parent)

        self.coupled_graph = coupled_graph
        self.redraw = 0

        # Create some controls
        try:
            self.mc = wx.media.MediaCtrl(self, style=wx.SIMPLE_BORDER, size=(350,350))
        
        except NotImplementedError:
            self.Destroy()
            raise

        self.Bind(wx.media.EVT_MEDIA_LOADED, self.OnMediaLoaded)

        load_file_button = wx.Button(self, -1, "Load File")
        self.Bind(wx.EVT_BUTTON, self.OnLoadFile, load_file_button)

        self.play_button = wx.Button(self, -1, "Play")
        self.Bind(wx.EVT_BUTTON, self.OnPlay, self.play_button)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_play_button, self.play_button)
        self.playing = False

        stop_button = wx.Button(self, -1, "Stop")
        self.Bind(wx.EVT_BUTTON, self.OnStop, stop_button)

        slider = wx.Slider(self, -1, 0, 0, 0)
        self.slider = slider
        slider.SetMinSize((150, -1))
        self.Bind(wx.EVT_SLIDER, self.OnSeek, slider)

        self.st_size = StaticText(self, -1, size=(100,-1))
        self.st_len  = StaticText(self, -1, size=(100,-1))
        self.st_pos  = StaticText(self, -1, size=(100,-1))

        # setup the layout
        sizer = wx.GridBagSizer(5,5)
        sizer.Add(self.mc, (0,1), span=(5,1))#, flag=wx.EXPAND)
        sizer.Add(load_file_button, (5,0))
        sizer.Add(self.play_button, (6,0))
        sizer.Add(stop_button, (7,0))
        sizer.Add(slider, (6,1), flag=wx.EXPAND)
        sizer.Add(self.st_size, (1, 5))
        sizer.Add(self.st_len,  (2, 5))
        sizer.Add(self.st_pos,  (3, 5))
        self.SetSizer(sizer)

        # wx.CallAfter(self.DoLoadFile, os.path.abspath("~/Desktop/Drd87_EPM_bgremoved_demo.avi"))
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer)
        self.timer.Start(100)

    def OnLoadFile(self, evt):
        dlg = wx.FileDialog(self, message="Choose a media file", defaultDir=os.getcwd(), defaultFile="", style=wx.FD_OPEN | wx.FD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.DoLoadFile(path)

        dlg.Destroy()

    def DoLoadFile(self, path):
        self.play_button.Disable()

        if not self.mc.Load(path):
            wx.MessageBox("Unable to load {}".format(path), "ERROR", wx.ICON_ERROR | wx.OK)
        else:
            self.mc.GetBestSize()
            self.GetSizer().Layout()
            self.slider.SetRange(0, self.mc.Length())

    def OnMediaLoaded(self, evt):
        self.play_button.Enable()

    def OnPlay(self, evt):
        if not self.mc.Play():
            wx.MessageBox("Unable to play video file!", "ERROR", wx.ICON_ERROR | wx.OK)
        else:
            self.mc.GetBestSize()
            self.GetSizer().Layout()
            self.slider.SetRange(0, self.mc.Length())
            if not self.playing and self.coupled_graph:
                self.mc.Play()
                self.playing = True
                self.coupled_graph.paused = False
            else:
                self.mc.Pause()
                self.playing = False
                self.coupled_graph.paused = True

            self.on_update_play_button(evt)

    def on_update_play_button(self, event):
        label = "Pause" if self.playing else "Play"
        self.play_button.SetLabel(label)

    # def OnPause(self, evt):
    #    self.mc.Pause()

    def OnStop(self, evt):
        self.mc.Stop()
        self.playing = False
        self.on_update_play_button(evt)

        if self.coupled_graph:
            self.coupled_graph.paused = True
            self.coupled_graph.reset_plot()

    def OnSeek(self, evt):
        offset = self.slider.GetValue()
        # print(offset)
        self.mc.Seek(offset)
        self.redraw += 1
        if self.coupled_graph and self.redraw == 1:
            self.redraw = 0
            self.coupled_graph.datagen.index = offset // 100
            self.coupled_graph.draw_plot()

    def OnTimer(self, evt):
        offset = self.mc.Tell()
        self.slider.SetValue(offset)
        self.st_size.SetLabel("size: {}".format(self.mc.GetBestSize()))
        self.st_len.SetLabel('length: %d seconds' % (self.mc.Length()/1000))
        self.st_pos.SetLabel('position: %d' % offset)

    def ShutdownDemo(self):
        self.timer.Stop()
        del self.timer

if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(None, size=(640,480))
    panel = VideoPanel(frame)
    frame.Show()
    app.MainLoop()
