# import external libraries
import wx # 2.8
import vlc
from client import Client
from server import Server

# import standard libraries
import os
import sys
import queue
import threading
import time

try:
    unicode        # Python 2
except NameError:
    unicode = str  # Python 3


class MiniPlayer(wx.Frame):
    """The main window has to deal with events.
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
        self.Bind(wx.EVT_MENU, self.OnOpen, id=1)
        self.Bind(wx.EVT_MENU, self.OnExit, id=2)
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
        self.Instance = vlc.Instance()
        self.player = self.Instance.media_player_new()

        self.OnOpen(None)
        self.data_q = q

        # Create the timer, which updates the video by reading from the queue
        self.timer = wx.Timer(self)
        self.timer.Start(95)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)

    def OnExit(self, evt):
        """Closes the window.
        """
        self.Close()

    def OnOpen(self, evt):
        """Pop up a new dialow window to choose a file, then play the selected file.
        """

        # Create a file dialog opened in the current home directory, where
        # you can display all kind of files, having as title "Choose a file".
        # dlg = wx.FileDialog(self, "Choose a file", os.path.expanduser('~'), "", "*.*", wx.OPEN)
        dlg = wx.FileDialog(self, message="Choose a media file", defaultDir=os.getcwd(), defaultFile="", style=wx.FD_OPEN | wx.FD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            dirname = dlg.GetDirectory()
            filename = dlg.GetFilename()
            # Creation
            self.Media = self.Instance.media_new(unicode(os.path.join(dirname, filename)))
            self.player.set_media(self.Media)

            # Report the title of the file chosen
            title = self.player.get_title()

            #  if an error was encountred while retriving the title, then use filename
            if title == -1:
                title = filename

            self.SetTitle("{}".format(title))

            # set the window id where to render VLC's video output
            handle = self.videopanel.GetHandle()
            if sys.platform.startswith("linux"): # for Linux using the X Server
                self.player.set_xwindow(handle)
            elif sys.platform == "win32": # for Windows
                self.player.set_hwnd(handle)
            elif sys.platform == "darwin": # for MacOS
                self.player.set_nsobject(handle)

            # Start playing the video as soon as it loads.
            self.player.play()

        # finally destroy the dialog
        dlg.Destroy()

    def OnTimer(self, evt):
        try:
            curr_time = self.data_q.get(block=False)
            if not self.player.is_playing():
                self.player.set_time(curr_time*100)
                self.player.play()
        except:
            if self.player.is_playing():
                self.player.pause()
            return


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
