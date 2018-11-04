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
from dialog import MyDialog

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

        self.prompt_for_data_selection()
        self.parse_data_selection()
        print(self.neurons)
        print(self.behaviors)
        print(type(self.behaviors))
        self.dataset.fillna(0)
        self.neuron_col_vectors = self.dataset[self.neurons]

        self.index = 0
        self.global_ymax = self.neuron_col_vectors.max().max()
        self.all_behavior_intervals = self.get_behavior(self.dataset)
        del self.dataset

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

    def prompt_for_data_selection(self):
        dlg = MyDialog(None, "Dialog")

        if dlg.ShowModal() == wx.ID_OK:
            user_input = dlg.get_user_input()
            self.neurons, self.behaviors = user_input

        dlg.Destroy()

    def parse_data_selection(self):
        if "range" in self.neurons:
            begin = int(self.neurons["range"][0])
            end = int(self.neurons["range"][0])
            self.neurons = [str(i) for i in range(begin, end+1)]
        elif "custom" in self.neurons:
            self.neurons = self.neurons["custom"]
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

class BoundControlBox(wx.Panel):
    """A static box with radio buttons and a text box. Allows to switch between
       automatic mode and manual mode (with an associated value).
    """
    def __init__(self, parent, ID, label, initval):
        wx.Panel.__init__(self, parent, ID)

        self.value = initval
        self.auto_radio_button_state = True
        self.manual_radio_button_state = False

        box = wx.StaticBox(self, -1, label)
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)

        self.radio_auto = wx.RadioButton(self, -1, label="Auto", style=wx.RB_GROUP)
        self.radio_auto.SetValue(True)

        self.radio_manual = wx.RadioButton(self, -1, label="Manual")
        self.radio_manual.SetValue(False)

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
        self.manual_radio_button_state = False

    def is_auto(self):
        return self.radio_auto.GetValue()

    def manual_value(self):
        return self.value

    def state_changed(self):
        if self.auto_radio_button_state != self.radio_auto.GetValue():
            self.auto_radio_button_state = self.radio_auto.GetValue()
            return True
        if not self.manual_radio_button_state:
            self.manual_radio_button_state = True
            return True
        else:
            return False

class GraphPanel(wx.Panel):
    """The main frame of the application"""

    def __init__(self, parent, coupled=False):
        wx.Panel.__init__(self, parent=parent, size=wx.Size(800, 800))

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

        if not self.coupled:
            self.pause_button = wx.Button(self, -1, "Pause")
            self.Bind(wx.EVT_BUTTON, self.on_pause_button, self.pause_button)
            self.Bind(wx.EVT_UPDATE_UI, self.on_update_pause_button, self.pause_button)


        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox2.Add(self.xmin_control, border=3, flag=wx.ALL)
        self.hbox2.Add(self.xmax_control, border=3, flag=wx.ALL)
        self.hbox2.AddSpacer(10)
        self.hbox2.Add(self.ymax_control, border=3, flag=wx.ALL)

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.hbox2, 0, flag=wx.ALIGN_LEFT | wx.TOP)

        self.SetSizer(self.vbox)
        self.vbox.Fit(self)

    # def create_status_bar(self):
      #  self.statusbar = self.CreateStatusBar()

    def init_plot(self):
        self.dpi = 400
        self.fig = Figure((3.0, 3.0), dpi=self.dpi)
        self.axes = self.fig.subplots(len(self.datagen.neurons), 1)
        self.fig.subplots_adjust(left=0.02, bottom=0.0, right=0.98, top=1.0, hspace=0.0)
        background_colors = ["red", "orange", "blue", "green"]
        self.plots_data = []

        for i, ax in enumerate(self.axes):

            # Plot data as line series, and save reference to the plotted line series.
            ax.plot(self.datagen.neuron_col_vectors[str(self.datagen.neurons[i])], linewidth=0.5)
            self.plots_data.append(ax.plot([0], linewidth=0.5, color="red")[0])
            pylab.setp(ax.get_xticklabels(), fontsize=3)
            pylab.setp(ax.get_yticklabels(), fontsize=2)

            for i, behavior_intervals in enumerate(self.datagen.all_behavior_intervals):
                for interval in behavior_intervals:
                    ax.axvspan(interval[0], interval[-1], alpha=0.1, color=background_colors[i])

            ax.axis("off")

        for plot_data in self.plots_data:
            plot_data.set_data([self.datagen.index, self.datagen.index], [0, 400])

    def bounds_changed(self):
        if self.xmin_control.state_changed():
            return True
        elif self.xmax_control.state_changed():
            return True
        elif self.ymax_control.state_changed():
            return True
        else:
            return False

    def set_new_bounds(self):
        if self.bounds_changed():

            if self.xmin_control.is_auto():
                xmin = 0
            else:
                xmin = int(self.xmin_control.manual_value())

            if self.xmax_control.is_auto():
                xmax = len(self.datagen.neuron_col_vectors.index)
            else:
                xmax = int(self.xmax_control.manual_value())

            if self.ymax_control.is_auto():
                ymax = self.datagen.global_ymax
            else:
                ymax = int(self.ymax_control.manual_value())

            # TEST
            print("set_new_bounds was called!, xmin={} xmax={} ymax={}".format(xmin, xmax, ymax))

            for i, ax in enumerate(self.axes):
                ax.set_xbound(lower=xmin, upper=xmax)
                ax.set_ybound(lower=-1, upper=ymax)

    def draw_plot(self):
        """Redraws the plot"""
        self.set_new_bounds()

        for plot in self.plots_data:
            plot.set_data([self.datagen.index, self.datagen.index], [0, 400])

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
