"""This module contains a bare-bones vlc player class to stream videos.
"""

import os
import queue
import platform

import wx
import vlc
from client import Client

class MiniPlayer(wx.Frame):
    """Stripped-down PyVLC video player class to sync with a main video player.
    """

    def __init__(self, title, q):
        wx.Frame.__init__(self, None, -1, title, pos=wx.DefaultPosition, size=(550, 500))

        # Menu Bar
        # File Menu
        self.frame_menubar = wx.MenuBar()
        self.file_menu = wx.Menu()
        self.file_menu.Append(1, "&Open", "Open from file..")
        self.file_menu.AppendSeparator()
        self.file_menu.Append(2, "&Close", "Quit")
        self.file_menu.AppendSeparator()
        self.Bind(wx.EVT_MENU, self.on_open, id=1)
        self.Bind(wx.EVT_MENU, self.on_exit, id=2)
        self.frame_menubar.Append(self.file_menu, "File")
        self.SetMenuBar(self.frame_menubar)

        # Panels
        # The first panel holds the video and it's all black
        self.videopanel = wx.Panel(self, -1)
        self.videopanel.SetBackgroundColour(wx.BLACK)

        # Put everything together
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.videopanel, 1, flag=wx.EXPAND)
        self.SetSizer(sizer)
        self.SetMinSize((350, 300))

        # VLC player controls
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

        self.on_open(None)
        self.data_q = q

        # Create the timer, which updates the video by reading from the queue
        self.timer = wx.Timer(self)
        self.timer.Start(100)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)

        try:
            self.current_time = self.data_q.get(block=False)
            self.current_time *= 100
        except:
            self.current_time = 0

    def on_exit(self, evt):
        """Closes the window.
        """
        self.Close()

    def on_open(self, evt):
        """Pop up a new dialow window to choose a file, then play the selected file.
        """

        # Create a file dialog opened in the current home directory, where
        # you can display all kind of files, having as title "Choose a file".
        # dlg = wx.FileDialog(self, "Choose a file", os.path.expanduser('~'), "", "*.*", wx.OPEN)
        dlg = wx.FileDialog(self, message="Choose a media file", defaultDir=os.getcwd(), defaultFile="", style=wx.FD_OPEN | wx.FD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            dirname = dlg.GetDirectory()
            filename = dlg.GetFilename()

            # Create
            self.media = self.instance.media_new(os.path.join(dirname, filename))
            self.player.set_media(self.media)

            # Report the title of the file chosen
            title = self.player.get_title()

            #  if an error was encountred while retriving the title, then use filename
            if title == -1:
                title = filename

            self.SetTitle("{}".format(title))

            # set the window id where to render VLC's video output
            handle = self.videopanel.GetHandle()
            if platform.system() == "Linux":
                self.player.set_xwindow(handle)
            elif platform.system() == "Windows":
                self.player.set_hwnd(handle)
            elif platform.system() == "Darwin":
                self.player.set_nsobject(handle)

            # Start playing the video as soon as it loads.
            self.player.play()

        # finally destroy the dialog
        dlg.Destroy()

    def on_timer(self, evt):
        try:
            current_time = self.data_q.get(block=False)
            if current_time != self.current_time:
                self.current_time = current_time
                update = True
            else:
                update = False
        except:
            if self.player.is_playing():
                self.player.pause()
            return

        if not self.player.is_playing():
            self.player.set_time(self.current_time*100)
            self.player.play()
        elif update:
            self.player.set_time(self.current_time*100)


if __name__ == "__main__":

    # Create a wx.App(), which handles the windowing system event loop
    app = wx.App()

    q = queue.Queue()
    client = Client("localhost", 10000, q)

    # Create the window containing our small media player
    player = MiniPlayer("Simple PyVLC Player", q)

    # Show the player window centred and run the application
    player.Centre()
    player.Show()
    app.MainLoop()
