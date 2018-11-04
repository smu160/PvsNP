import wx

class MyDialog(wx.Dialog):

    def __init__(self, parent, title):
        super(MyDialog, self).__init__(parent, title=title, size=(600, 250))

        pnl = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        sb = wx.StaticBox(pnl, label="Neurons and Behavior")
        sbs = wx.StaticBoxSizer(sb, orient=wx.VERTICAL)

        hbox0 = wx.BoxSizer(wx.HORIZONTAL)
        self.range_radio_button = wx.RadioButton(pnl, label="From: ", style=wx.RB_GROUP)

        self.from_text_box = wx.TextCtrl(self, -1, size=(35, -1), value="1", style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_range, self.from_text_box)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_neurons_enter, self.from_text_box)

        to_text = wx.StaticText(pnl, wx.ID_ANY, label="to:", style=wx.ALIGN_CENTER)

        self.to_text_box = wx.TextCtrl(self, -1, size=(35, -1), value="10", style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_neurons_enter, self.to_text_box)

        hbox0.Add(self.range_radio_button, flag=wx.BOTTOM, border=10)
        hbox0.Add(self.from_text_box, flag=wx.LEFT, border=5)
        hbox0.Add(to_text, flag=wx.LEFT, border=5)
        hbox0.Add(self.to_text_box, flag=wx.LEFT, border=5)
        sbs.Add(hbox0)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.custom_radio_button = wx.RadioButton(pnl, label="Custom range: ")
        hbox1.Add(self.custom_radio_button)

        self.custom_range = wx.TextCtrl(pnl, value="1,3,5,7,8", style=wx.TE_PROCESS_ENTER, size=(200, -1))
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_range, self.custom_range)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_neurons_enter, self.custom_range)

        hbox1.Add(self.custom_range, flag=wx.LEFT, border=5)
        sbs.Add(hbox1)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(wx.StaticText(pnl, wx.ID_ANY, label="Behaviors: "))
        self.behaviors_text = wx.TextCtrl(pnl, value="OpenArms_centerpoint, ClosedArms_centerpoint", style=wx.TE_PROCESS_ENTER, size=(600, -1))
        self.behaviors_text.Bind(wx.EVT_SET_FOCUS, self.on_custom_focus)
        self.behaviors_text.Bind(wx.EVT_TEXT_ENTER, self.on_behaviors_enter)

        hbox3.Add(self.behaviors_text, flag=wx.LEFT, border=5)
        sbs.Add(hbox3, flag=wx.TOP, border=50)

        pnl.SetSizer(sbs)

        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        done_button = wx.Button(self, wx.ID_OK, label="Done")
        hbox4.Add(done_button)

        vbox.Add(pnl, proportion=1, flag=wx.ALL|wx.EXPAND, border=5)
        vbox.Add(hbox4, flag=wx.ALIGN_CENTER|wx.TOP|wx.BOTTOM, border=10)

        self.SetSizer(vbox)

        self.neurons = None
        self.behaviors = None

    def on_update_range(self, event):
        self.from_text_box.Enable(self.range_radio_button.GetValue())
        self.to_text_box.Enable(self.range_radio_button.GetValue())
        self.custom_range.Enable(self.custom_radio_button.GetValue())

    def on_neurons_enter(self, event):
        if self.range_radio_button.GetValue():
            self.neurons = {"range": (self.from_text_box.GetValue(), self.to_text_box.GetValue())}
        elif self.custom_radio_button.GetValue():
            self.neurons = {"custom": self.custom_range.GetValue()}

    def on_custom_focus(self, event):
        self.behaviors_text.SetValue("")

    def on_behaviors_enter(self, event):
        self.behaviors = self.behaviors_text.GetValue()

    def get_user_input(self):
        return self.neurons, self.behaviors

class MyWin(wx.Frame):

    def __init__(self, parent, title):
        super(MyWin, self).__init__(parent, title = title, size = (250,150))
        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        self.on_modal()
        self.Centre()
        self.Show(True)

    def on_modal(self):
        dlg = MyDialog(self, "Dialog")
        if dlg.ShowModal() == wx.ID_OK:
            user_input = dlg.get_user_input()
            print(user_input)
    
        dlg.Destroy()

if __name__ == "__main__":
    ex  =  wx.App()
    MyWin(None, "MenuBar demo")
    ex.MainLoop()
