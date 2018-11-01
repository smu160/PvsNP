import os
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas, NavigationToolbar2WxAgg as NavigationToolbar
import pylab
import numpy as np
import wx
import pandas as pd
import matplotlib
matplotlib.use('WXAgg')

class DataGen(object):
    """Data Generator/Extractor class"""

    def __init__(self, init=0):
        self.file_path = ""

        while not self.file_path:
            wx.MessageBox("Select a data file", "Info", wx.OK | wx.ICON_INFORMATION)
            self.get_file_path()

        try:
            self.dataset = pd.read_csv(self.file_path)
        except Exception:
            wx.MessageBox("Unable to load {}".format(self.file_path), "ERROR", wx.ICON_ERROR | wx.OK)
            sys.exit(1)

        self.dataset.fillna(0)
        self.index = 0
        self.all_behavior_intervals = self.get_behavior(self.dataset)

    def get_file_path(self):
        file_dialog = wx.FileDialog(None, message="Select directory to open", defaultDir=os.getcwd(), defaultFile="", style=wx.FD_OPEN | wx.FD_CHANGE_DIR)

        # This function returns the button pressed to close the dialog.
        # Let's check if user clicked OK or pressed ENTER
        if file_dialog.ShowModal() == wx.ID_OK:
            print("You selected: {}".format(file_dialog.GetPath()))
            self.file_path = file_dialog.GetPath()
        else:
            print("You clicked cancel")

        # The dialog is not in the screen anymore, but it's still in memory
        file_dialog.Destroy()

    def next(self):
        self.index += 1
        return self.index

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
        self.paused = True

        # self.create_menu()
        # self.create_status_bar()
        self.create_main_panel()

        if not self.coupled:
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
        self.xmax_control = BoundControlBox(self, -1, "X max", 1000)
        self.ymax_control = BoundControlBox(self, -1, "Y max", 200)
        # self.window_width = BoundControlBox(self, -1, "Window width", 0)

        if not self.coupled:
            self.pause_button = wx.Button(self, -1, "Pause")
            self.Bind(wx.EVT_BUTTON, self.on_pause_button, self.pause_button)
            self.Bind(wx.EVT_UPDATE_UI, self.on_update_pause_button, self.pause_button)

        # self.cb_grid = wx.CheckBox(self, -1, "Show Grid", style=wx.ALIGN_RIGHT)
        # self.Bind(wx.EVT_CHECKBOX, self.on_cb_grid, self.cb_grid)
        # self.cb_grid.SetValue(False)

        # self.cb_xlab = wx.CheckBox(self, -1, "Show X labels", style=wx.ALIGN_RIGHT)
        # self.Bind(wx.EVT_CHECKBOX, self.on_cb_xlab, self.cb_xlab)
        # self.cb_xlab.SetValue(True)

        # self.load_file_button = wx.Button(self, -1, "Load Data File")
        # self.Bind(wx.EVT_BUTTON, self.on_load_file, self.load_file_button)

        # self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        # self.hbox1.AddSpacer(20)
        # self.hbox1.Add(self.cb_grid, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        # self.hbox1.AddSpacer(10)
        # self.hbox1.Add(self.cb_xlab, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)

        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox2.Add(self.xmin_control, border=3, flag=wx.ALL)
        self.hbox2.Add(self.xmax_control, border=3, flag=wx.ALL)
        self.hbox2.AddSpacer(10)
        self.hbox2.Add(self.ymax_control, border=3, flag=wx.ALL)
        # self.hbox2.Add(self.window_width, border=3, flag=wx.ALL)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP | wx.GROW)
        # self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.vbox.Add(self.hbox2, 0, flag=wx.ALIGN_LEFT | wx.TOP)

        self.SetSizer(self.vbox)
        self.vbox.Fit(self)

    # def create_status_bar(self):
      #  self.statusbar = self.CreateStatusBar()

    def init_plot(self):
        self.dpi = 400
        self.fig = Figure((3.0, 3.0), dpi=self.dpi)
        self.axes = self.fig.subplots(10, 1)
        self.fig.subplots_adjust(hspace=0.1)
        background_colors = ["red", "orange", "cyan", "grey"]
        self.plots_data = []

        for i, ax in enumerate(self.axes):

            # Plot data as line series, and save reference to the plotted line series.
            ax.plot(self.datagen.dataset[str(i+1)], linewidth=0.5)
            self.plots_data.append(ax.plot([0], linewidth=0.5, color="red")[0])
            pylab.setp(ax.get_xticklabels(), fontsize=3)
            pylab.setp(ax.get_yticklabels(), fontsize=2)

            for i, behavior_intervals in enumerate(self.datagen.all_behavior_intervals):
                for interval in behavior_intervals:
                    ax.axvspan(interval[0], interval[-1], alpha=0.1, color=background_colors[i])

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        for plot_data in self.plots_data:
            plot_data.set_data([self.datagen.index, self.datagen.index], [0, 400])

    def draw_plot(self):
        """Redraws the plot"""

        # When xmin is on auto, it "follows" xmax to produce a sliding window
        # effect. therefore, xmin is assigned after xmax.
        if self.xmax_control.is_auto():
            xmax = len(self.datagen.dataset.index) # self.datagen.index if self.datagen.index > window_width else window_width
        else:
            xmax = int(self.xmax_control.manual_value())

        if self.xmin_control.is_auto():
            xmin = 0
        else:
            xmin = int(self.xmin_control.manual_value())

        # for ymin and ymax, find the minimal and maximal values
        # in the data set and add a mininal margin.
        # Note: it's easy to change this scheme to the minimal/maximal value
        # in the current display, and not the whole data set.
        if self.ymax_control.is_auto():
            ymax = self.datagen.dataset[[str(i+1) for i, _ in enumerate(self.axes)]].max().max()
        else:
            ymax = int(self.ymax_control.manual_value())

        for i, ax in enumerate(self.axes):
            ax.set_xbound(lower=xmin, upper=xmax)
            ax.set_ybound(lower=-1, upper=ymax)

        # if self.cb_grid.IsChecked():
        #    self.axes1.grid(True, color='gray')
        # else:
        #    self.axes1.grid(False)

        # pylab.setp(self.axes1.get_xticklabels(), visible=self.cb_xlab.IsChecked())
        # self.plot_data.set_xdata(np.arange(len(self.data)))
        # self.plot_data.set_ydata(np.array(self.data))
        for plot_data in self.plots_data:
            plot_data.set_data([self.datagen.index, self.datagen.index], [0, 400])

        self.canvas.draw()

        # Make sure that the GUI framework has a chance to run its event loop &
        # clear any GUI events. This needs to be in a try/except block since the
        # default implementation of this method is to raise NotImplementedError
        try:
            self.canvas.flush_events()
        except NotImplementedError:
            pass

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

    def on_redraw_timer(self, new_index):
        # if paused do not add data, but still redraw the plot
        # (to respond to scale modifications, grid change, etc.)
        if not self.paused:
            self.datagen.index = new_index

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
    app = wx.App(False)
    frame = wx.Frame(None, size=(1024, 768))
    panel = GraphPanel(frame)
    frame.Show()
    app.MainLoop()
