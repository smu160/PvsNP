import wx
import wx.media
import os

import matplotlib       # Provides the graph figures
matplotlib.use('WXAgg') # matplotlib needs a GUI (layout), we use wxPython
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas
from multiprocessing import Process

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

    def __init__(self, parent, coupled_graph=None, timer=None):
        wx.Panel.__init__(self, parent=parent)
        self.coupled_graph = coupled_graph

        # Create some controls
        try:
            self.mc = wx.media.MediaCtrl(self, style=wx.BORDER_THEME, size=(400,400))
            self.mc2 = wx.media.MediaCtrl(self, style=wx.SIMPLE_BORDER, size=(400,400))
        except NotImplementedError:
            self.Destroy()
            raise

        self.Bind(wx.media.EVT_MEDIA_LOADED, self.OnMediaLoaded)

        load_vid1_button = wx.Button(self, -1, "Load Video 1")
        self.Bind(wx.EVT_BUTTON, self.OnLoadFile, load_vid1_button)
        load_vid2_button = wx.Button(self, -1, "Load Video 2")
        self.Bind(wx.EVT_BUTTON, self.OnLoadFile, load_vid2_button)

        self.play_button = wx.Button(self, -1, "Play")
        self.Bind(wx.EVT_BUTTON, self.OnPlay, self.play_button)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_play_button, self.play_button)
        self.playing = False

        stop_button = wx.Button(self, -1, "Stop")
        self.Bind(wx.EVT_BUTTON, self.OnStop, stop_button)

        slider = wx.Slider(self, -1, 0, 0, 0)
        self.slider = slider
        slider.SetMinSize((250, -1))
        self.Bind(wx.EVT_SLIDER, self.OnSeek, slider)
        
        # Setup the layout
        sizer = wx.GridBagSizer(0, 0)
        sizer.Add(self.mc, (1, 0), span=(5, 5)) #, flag=wx.EXPAND)
        sizer.Add(self.mc2, (9, 0), span=(5, 5))
        sizer.Add(load_vid1_button, (0, 1))
        sizer.Add(load_vid2_button, (8, 1))
        sizer.Add(self.play_button, (6, 0))
        sizer.Add(stop_button, (7, 0))
        sizer.Add(slider, (6, 1), flag=wx.EXPAND)
        sizer.AddGrowableCol(0, 0)
        self.SetSizer(sizer)

        self.init_contours()

        if timer:
            self.timer = timer
        else:
            self.timer = wx.Timer(self)

        self.Bind(wx.EVT_TIMER, self.OnTimer)
        self.timer.Start(100)

    def OnLoadFile(self, evt):
        button = evt.GetEventObject().GetLabel()
        dlg = wx.FileDialog(self, message="Choose a media file", defaultDir=os.getcwd(), defaultFile="", style=wx.FD_OPEN | wx.FD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            if button == "Load Video 1":
                self.DoLoadFile(path, self.mc)
            elif button == "Load Video 2":
                self.DoLoadFile(path, self.mc2)

        dlg.Destroy()

    def DoLoadFile(self, path, media_ctrl):
        self.play_button.Disable()

        if not media_ctrl.Load(path):
            wx.MessageBox("Unable to load {}".format(path), "ERROR", wx.ICON_ERROR | wx.OK)
        else:
            media_ctrl.GetBestSize()
            self.GetSizer().Layout()
            self.slider.SetRange(0, media_ctrl.Length())

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
                self.mc2.Play()
                self.playing = True
                self.coupled_graph.paused = False
            else:
                self.mc.Pause()
                self.mc2.Pause()
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
        self.mc2.Stop()
        self.playing = False
        self.on_update_play_button(evt)

        if self.coupled_graph:
            self.coupled_graph.paused = True
            self.coupled_graph.reset_plot()

    def OnSeek(self, evt):
        offset = self.slider.GetValue()
        self.mc.Seek(offset)
        self.mc2.Seek(offset)
        if self.coupled_graph:
           self.coupled_graph.datagen.index = offset // 100
           # self.coupled_graph.draw_plot()

    def OnTimer(self, evt):
        offset = self.mc.Tell()
        self.slider.SetValue(offset)
        if self.coupled_graph:
            self.coupled_graph.on_redraw_timer(offset//100)
            self.coupled_graph.datagen.index = offset // 100

    def ShutdownDemo(self):
        self.timer.Stop()
        del self.timer

    def init_contours(self):
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Centre()
        self.Show(True)

    def on_paint(self, e):
        dc = wx.PaintDC(self)

        # Red non-filled circle
        dc.SetPen(wx.Pen("red"))
        dc.SetBrush(wx.Brush("red", wx.TRANSPARENT))
        dc.DrawCircle(50, 100, 10)


if __name__ == "__main__":
    app = wx.App(False)
    frame = wx.Frame(None, size=(640,480))
    panel = VideoPanel(frame)
    frame.Show()
    app.MainLoop()
