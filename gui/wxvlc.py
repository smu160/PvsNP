#! /usr/bin/python
# -*- coding: utf-8 -*-

#
# WX example for VLC Python bindings
# Copyright (C) 2009-2010 the VideoLAN team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
#
"""
A simple example for VLC python bindings using wxPython.

Author: Michele Orrù
Date: 23-11-2010
"""

# import standard libraries
import os
import sys
import platform
import queue
import time
import subprocess

# import external libraries
import wx # 2.8
import vlc
from server import Server

try:
    unicode        # Python 2
except NameError:
    unicode = str  # Python 3


class Player(wx.Frame):
    """The main window has to deal with events.
    """

    def __init__(self, title):
        wx.Frame.__init__(self, None, -1, title, pos=wx.DefaultPosition, size=(550, 500))

        # Menu Bar
        # File Menu
        self.frame_menubar = wx.MenuBar()
        self.file_menu = wx.Menu()
        self.file_menu.Append(1, "&Open vid", "Open from file..")
        self.file_menu.AppendSeparator()
        self.file_menu.Append(2, "&New Plot", "Create new plot..")
        self.file_menu.AppendSeparator()
        self.file_menu.Append(3, "&Add Video", "Add another vid..")
        self.file_menu.AppendSeparator()
        self.file_menu.Append(4, "&Close", "Quit")
        self.Bind(wx.EVT_MENU, self.OnOpen, id=1)
        self.Bind(wx.EVT_MENU, self.OnNewPlot, id=2)
        self.Bind(wx.EVT_MENU, self.OnStream, id=3)
        self.Bind(wx.EVT_MENU, self.OnExit, id=4)
        self.frame_menubar.Append(self.file_menu, "File")
        self.SetMenuBar(self.frame_menubar)

        # Panels
        # The first panel holds the video and it's all black
        self.videopanel = wx.Panel(self, -1)
        self.videopanel.SetBackgroundColour(wx.BLACK)

        # The second panel holds controls
        ctrlpanel = wx.Panel(self, -1)
        self.timeslider = wx.Slider(ctrlpanel, -1, 0, 0, 1000)
        self.timeslider.SetRange(0, 1000)
        pause = wx.Button(ctrlpanel, label="Pause")
        play = wx.Button(ctrlpanel, label="Play")
        stop = wx.Button(ctrlpanel, label="Stop")
        volume = wx.Button(ctrlpanel, label="Volume")
        self.volslider = wx.Slider(ctrlpanel, -1, 0, 0, 100, size=(100, -1))

        # Bind controls to events
        self.Bind(wx.EVT_SLIDER, self.OnSeek, self.timeslider)
        self.Bind(wx.EVT_BUTTON, self.OnPlay, play)
        self.Bind(wx.EVT_BUTTON, self.OnPause, pause)
        self.Bind(wx.EVT_BUTTON, self.OnStop, stop)
        self.Bind(wx.EVT_BUTTON, self.OnToggleVolume, volume)
        self.Bind(wx.EVT_SLIDER, self.OnSetVolume, self.volslider)

        # Give a pretty layout to the controls
        ctrlbox = wx.BoxSizer(wx.VERTICAL)
        box1 = wx.BoxSizer(wx.HORIZONTAL)
        box2 = wx.BoxSizer(wx.HORIZONTAL)

        # box1 contains the timeslider
        box1.Add(self.timeslider, 1)

        # box2 contains some buttons and the volume controls
        box2.Add(play, flag=wx.RIGHT, border=5)
        box2.Add(pause)
        box2.Add(stop)
        box2.Add((-1, -1), 1)
        box2.Add(volume)
        box2.Add(self.volslider, flag=wx.TOP | wx.LEFT, border=5)

        # Merge box1 and box2 to the ctrlsizer
        ctrlbox.Add(box1, flag=wx.EXPAND | wx.BOTTOM, border=10)
        ctrlbox.Add(box2, 1, wx.EXPAND)
        ctrlpanel.SetSizer(ctrlbox)

        # Put everything together
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.videopanel, 1, flag=wx.EXPAND)
        sizer.Add(ctrlpanel, flag=wx.EXPAND | wx.BOTTOM | wx.TOP, border=10)
        self.SetSizer(sizer)
        self.SetMinSize((350, 300))

        # finally create the timer, which updates the timeslider
        self.timer = wx.Timer(self)
        self.timer.Start(100)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)

        # VLC player controls
        self.Instance = vlc.Instance()
        self.player = self.Instance.media_player_new()

        self.q = queue.Queue()

    def OnStream(self, evt):
        if platform.system() == "Darwin":
            subprocess.Popen(["pythonw", "mini_player.py"])
        else:
            subprocess.Popen(["python", "mini_player.py"])

    def OnNewPlot(self, evt):
        subprocess.Popen(["python", "client.py"])

    def OnExit(self, evt):
        """Closes the window."""
        self.Close()

    def OnOpen(self, evt):
        """Pop up a new dialow window to choose a file, then play the selected file.
        """
        # if a file is already running, then stop it.
        self.OnStop(None)

        # Create a file dialog opened in the current home directory, where you
        # can display all kind of files, having as title "Choose a file".
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

            # Set the volume slider to the current volume
            self.volslider.SetValue(self.player.audio_get_volume() / 2)

        # Finally, destroy the dialog
        dlg.Destroy()

    def OnPlay(self, evt):
        """Toggle the status to Play/Pause.

        If no file is loaded, open the dialog window.
        """

        # Check if there is a file to play, otherwise open a
        # wx.FileDialog to select a file
        if not self.player.get_media():
            self.OnOpen(None)
        else:
            # Try to launch the media, if this fails display an error message
            if self.player.play() == -1:
                self.errorDialog("Unable to play.")
            else:
                self.timer.Start()

    def OnPause(self, evt):
        """Pause the player."""
        self.player.pause()

    def OnStop(self, evt):
        """Stop the player."""
        self.player.stop()

        # reset the queue
        self.q.queue.clear()
        self.q.put('d')
        self.q.put('0')

        # reset the time slider
        self.timeslider.SetValue(0)
        self.timer.Stop()

    def OnSeek(self, evt):
        self.timer.Stop()
        offset = self.timeslider.GetValue()

        if offset >= 0:
            self.q.queue.clear()
            self.q.put("d")
            curr_time = self.player.get_time()
            curr_time //= 100
            time.sleep(0.005)
            self.q.put(curr_time)

        self.player.set_position(offset/1000)
        self.timer.Start(100)

    def OnTimer(self, evt):
        """Update the time slider according to the current movie time."""

        # Update the time on the slider
        pos = self.player.get_position()

        if pos >= 0 and self.player.is_playing():
            curr_time = self.player.get_time()
            curr_time //= 100
            self.q.put(curr_time)
        else:
            self.q.queue.clear()

        self.timeslider.SetValue(pos*1000)

    def OnToggleVolume(self, evt):
        """Mute/Unmute according to the audio button."""

        is_mute = self.player.audio_get_mute()
        self.player.audio_set_mute(not is_mute)

        # update the volume slider;
        # since vlc volume range is in [0, 200],
        # and our volume slider has range [0, 100], just divide by 2.
        self.volslider.SetValue(self.player.audio_get_volume() / 2)

    def OnSetVolume(self, evt):
        """Set the volume according to the volume sider.
        """
        volume = self.volslider.GetValue() * 2
        # vlc.MediaPlayer.audio_set_volume returns 0 if success, -1 otherwise

        if self.player.audio_set_volume(volume) == -1:
            self.errorDialog("Failed to set volume")

    def errorDialog(self, errormessage):
        """Display a simple error dialog."""
        edialog = wx.MessageDialog(self, errormessage, 'Error', wx.OK | wx.ICON_ERROR)
        edialog.ShowModal()


def main():

    # Create a wx.App(), which handles the windowing system event loop
    app = wx.App()

    # Create the window containing video player
    player = Player("Video Player")

    server = Server("127.0.0.1", 10000, player.q)

    # Show the video player window centred and run the application
    player.Centre()
    player.Show()
    app.MainLoop()

if __name__ == "__main__":
    main()