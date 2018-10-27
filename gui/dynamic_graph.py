import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas, NavigationToolbar2WxAgg as NavigationToolbar
import pylab
import numpy as np
import wx
import wx.media
import pandas as pd
import matplotlib
matplotlib.use('WXAgg')

class DataGen(object):
    """Data Generator/Extractor class"""

    def __init__(self, init=0):
        self.data = self.init = init
        dataset = pd.read_csv("~/Desktop/drd87_test_file.csv")
        dataset.fillna(0)
        self.y = dataset["1"].values
        self.index = 0
        self.all_behavior_intervals = self.get_behavior(dataset) 

    def next(self):
        self._recalc_data()
        return self.data

    def _recalc_data(self):
        # self.data = self.y[self.index]
        self.index += 1

    def get_behavior(self, dataset):
        head_dips = self.extract_epochs(dataset, "Head_Dips")
        head_dip_intervals = self.filter_epochs(head_dips[1], framerate=1, seconds=1)

        open_arms = self.extract_epochs(dataset, "OpenArms_centerpoint")
        open_intervals = self.filter_epochs(open_arms[1], framerate=1, seconds=1)

        closed_arms = self.extract_epochs(dataset, "ClosedArms_centerpoint")
        closed_intervals = self.filter_epochs(closed_arms[1], framerate=1, seconds=1)

        center_epochs = self.extract_epochs(dataset, "Center")
        center_intervals = self.filter_epochs(center_epochs[1], framerate=1, seconds=1)
    
        all_behavior_intervals = [center_intervals, open_intervals, closed_intervals, head_dip_intervals]
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

class BoundControlBox(wx.Panel):
    """A static box with radio buttons and a text box. Allows to switch between
       automatic mode and manual mode (with an associated value).
    """
    def __init__(self, parent, ID, label, initval):
        wx.Panel.__init__(self, parent, ID)

        self.value = initval

        box = wx.StaticBox(self, -1, label)
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        self.radio_auto = wx.RadioButton(self, -1, label="Auto", style=wx.RB_GROUP)
        self.radio_manual = wx.RadioButton(self, -1, label="Manual")
        self.manual_text = wx.TextCtrl(self, -1, size=(35, -1), value=str(initval), style=wx.TE_PROCESS_ENTER)

        self.Bind(wx.EVT_UPDATE_UI, self.on_update_manual_text, self.manual_text)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_text_enter, self.manual_text)

        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        manual_box.Add(self.radio_manual, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.manual_text, flag=wx.ALIGN_CENTER_VERTICAL)

        sizer.Add(self.radio_auto, 0, wx.ALL, 10)
        sizer.Add(manual_box, 0, wx.ALL, 10)
        self.SetSizer(sizer)
        sizer.Fit(self)

    def on_update_manual_text(self, event):
        self.manual_text.Enable(self.radio_manual.GetValue())

    def on_text_enter(self, event):
        self.value = self.manual_text.GetValue()

    def is_auto(self):
        return self.radio_auto.GetValue()

    def manual_value(self):
        return self.value


class GraphPanel(wx.Panel):
    """The main frame of the application"""

    def __init__(self, parent, coupled=False):
        wx.Panel.__init__(self, parent=parent)

        self.coupled = coupled

        self.datagen = DataGen()
        self.data = [self.datagen.next()]
        self.paused = True

        # self.create_menu()
        # self.create_status_bar()
        self.create_main_panel()

        self.redraw_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_redraw_timer, self.redraw_timer)
        self.redraw_timer.Start(100)

    def create_menu(self):
        self.menubar = wx.MenuBar()

        menu_file = wx.Menu()
        m_expt = menu_file.Append(-1, "&Save plot\tCtrl-S", "Save plot to file")
        self.Bind(wx.EVT_MENU, self.on_save_plot, m_expt)
        menu_file.AppendSeparator()
        m_exit = menu_file.Append(-1, "E&xit\tCtrl-X", "Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)

        self.menubar.Append(menu_file, "&File")
        self.SetMenuBar(self.menubar)

    def create_main_panel(self):
        self.init_plot()
        self.canvas = FigCanvas(self, -1, self.fig)

        self.xmin_control = BoundControlBox(self, -1, "X min", 0)
        self.xmax_control = BoundControlBox(self, -1, "X max", 200)
        self.ymin_control = BoundControlBox(self, -1, "Y min", 0)
        self.ymax_control = BoundControlBox(self, -1, "Y max", 200)
        self.window_width = wx.TextCtrl(self, -1, "Window Width")

        if not self.coupled:
            self.pause_button = wx.Button(self, -1, "Pause")
            self.Bind(wx.EVT_BUTTON, self.on_pause_button, self.pause_button)
            self.Bind(wx.EVT_UPDATE_UI, self.on_update_pause_button, self.pause_button)

        self.cb_grid = wx.CheckBox(self, -1, "Show Grid", style=wx.ALIGN_RIGHT)
        self.Bind(wx.EVT_CHECKBOX, self.on_cb_grid, self.cb_grid)
        self.cb_grid.SetValue(False)

        self.cb_xlab = wx.CheckBox(self, -1, "Show X labels", style=wx.ALIGN_RIGHT)
        self.Bind(wx.EVT_CHECKBOX, self.on_cb_xlab, self.cb_xlab)
        self.cb_xlab.SetValue(True)

        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox1.Add(self.window_width, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.AddSpacer(20)
        self.hbox1.Add(self.cb_grid, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.AddSpacer(10)
        self.hbox1.Add(self.cb_xlab, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)

        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox2.Add(self.xmin_control, border=5, flag=wx.ALL)
        self.hbox2.Add(self.xmax_control, border=5, flag=wx.ALL)
        self.hbox2.AddSpacer(24)
        self.hbox2.Add(self.ymin_control, border=5, flag=wx.ALL)
        self.hbox2.Add(self.ymax_control, border=5, flag=wx.ALL)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.vbox.Add(self.hbox2, 0, flag=wx.ALIGN_LEFT | wx.TOP)

        self.SetSizer(self.vbox)
        self.vbox.Fit(self)

    # def create_status_bar(self):
      #  self.statusbar = self.CreateStatusBar()

    def init_plot(self):
        self.dpi = 300
        self.fig = Figure((1.0, 1.0), dpi=self.dpi)

        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor("white")
        self.axes.set_title("Neuron 0", size=4)

        pylab.setp(self.axes.get_xticklabels(), fontsize=2)
        pylab.setp(self.axes.get_yticklabels(), fontsize=2)

        # Plot data as line series, and save reference to the plotted line series.
        self.axes.plot(self.datagen.y, linewidth=0.8)
        # self.axes.vlines(0, 0, 100, linewidth=0.8, color="black")
        self.plot_data = self.axes.plot([0], linewidth=0.8, color="red")[0]

        all_behavior_intervals = self.datagen.all_behavior_intervals
        background_colors = ["red", "orange", "cyan", "grey"]
        
        for i, behavior_intervals in enumerate(all_behavior_intervals):
            for interval in behavior_intervals:
                self.axes.axvspan(interval[0], interval[-1], alpha=0.1, color=background_colors[i])
        
        # self.axes.axvspan(20, 50, alpha=0.1, color="red")

    def draw_plot(self):
        """Redraws the plot"""

        # When xmin is on auto, it "follows" xmax to produce a sliding window
        # effect. therefore, xmin is assigned after xmax.
        if self.xmax_control.is_auto():
            xmax = len(self.data) if len(self.data) > 100 else 100
        else:
            xmax = int(self.xmax_control.manual_value())

        if self.xmin_control.is_auto():
            xmin = xmax - 100
        else:
            xmin = int(self.xmin_control.manual_value())

        # for ymin and ymax, find the minimal and maximal values
        # in the data set and add a mininal margin.
        # Note: it's easy to change this scheme to the minimal/maximal value
        # in the current display, and not the whole data set.
        if self.ymin_control.is_auto():
            ymin = round(min(self.data), 0) - 1
        else:
            ymin = int(self.ymin_control.manual_value())

        if self.ymax_control.is_auto():
            ymax = round(max(self.data), 0) + 1
        else:
            ymax = int(self.ymax_control.manual_value())

        self.axes.set_xbound(lower=xmin, upper=xmax)
        self.axes.set_ybound(lower=ymin, upper=ymax)

        # anecdote: axes.grid assumes b=True if any other flag is
        # given even if b is set to False.
        # so just passing the flag into the first statement won't
        # work.
        if self.cb_grid.IsChecked():
            self.axes.grid(True, color='gray')
        else:
            self.axes.grid(False)

        # Using setp here is convenient, because get_xticklabels returns a list
        # over which one needs to explicitly iterate, and setp already handles
        # this.
        pylab.setp(self.axes.get_xticklabels(), visible=self.cb_xlab.IsChecked())
        #self.plot_data.set_xdata(np.arange(len(self.data)))
        #self.plot_data.set_ydata(np.array(self.data))
        self.plot_data.set_data([self.datagen.index, self.datagen.index], [0, 100])
        self.canvas.draw()

    def on_pause_button(self, event):
        self.paused = not self.paused

    def on_update_pause_button(self, event):
        label = "Resume" if self.paused else "Pause"
        self.pause_button.SetLabel(label)

    def on_cb_grid(self, event):
        self.draw_plot()

    def on_cb_xlab(self, event):
        self.draw_plot()

    def on_save_plot(self, event):
        file_choices = "PNG (*.png)|*.png"

        dlg = wx.FileDialog(
            self,
            message="Save plot as...",
            defaultDir=os.getcwd(),
            defaultFile="plot.png",
            wildcard=file_choices,
            style=wx.SAVE)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message("Saved to {}".format(path))

    def on_redraw_timer(self, event):
        # if paused do not add data, but still redraw the plot
        # (to respond to scale modifications, grid change, etc.)
        if not self.paused:
            self.data.append(self.datagen.next())

        self.draw_plot()

    def on_exit(self, event):
        self.Destroy()

    def reset_plot(self):
        self.datagen.index = 0

    def flash_status_message(self, msg, flash_len_ms=1500):
        self.statusbar.SetStatusText(msg)
        self.timeroff = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_flash_status_off, self.timeroff)
        self.timeroff.Start(flash_len_ms, oneShot=True)

    def on_flash_status_off(self, event):
        self.statusbar.SetStatusText('')


if __name__ == "__main__":
    app = wx.App()
    # frame = wx.Frame(None, size=(800,600))
    app.frame = GraphFrame()
    #app.frame.Show()
    app.MainLoop()
