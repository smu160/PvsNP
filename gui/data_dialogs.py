"""
This module contains all the dialogs to present to a user.

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import sys

from PyQt5 import QtWidgets, QtCore, QtGui

class DataDialog(QtWidgets.QDialog):
    """A dialog for presenting and selecting columns from a csv"""

    def __init__(self, column_names, parent=None, checkbox=False):
        super(DataDialog, self).__init__(parent)

        self.layout = QtWidgets.QVBoxLayout()
        self.listWidget = QtWidgets.QListWidget()
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget.setGeometry(QtCore.QRect(10, 10, 211, 291))

        # Fill QList with column names that were passed in
        for column_name in column_names:
            item = QtWidgets.QListWidgetItem(column_name)
            self.listWidget.addItem(item)

        self.button_box = QtWidgets.QHBoxLayout()
        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.clicked.connect(self.handle_ok_button)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.handle_cancel_button)

        self.button_box.addWidget(self.ok_button)
        self.button_box.addWidget(self.cancel_button)

        self.listWidget.itemSelectionChanged.connect(self.update_selected_items)

        self.layout.addWidget(self.listWidget)
        self.layout.addLayout(self.button_box)

        # Add a checkbox to the dialog for the user to toggle
        # if they want to customize behavior colors
        if checkbox:
            self.choose_colors_cb = QtWidgets.QCheckBox("Choose behavior colors", self)
            self.layout.addWidget(self.choose_colors_cb)
            self.choose_colors_cb.stateChanged.connect(self.handle_checkbox)

            self.choose_colors = False

        self.setLayout(self.layout)

        # Type of exit, (i.e. close, ok, cancel)
        # close: 0; cancel: 0; ok: 1
        self.exit_status = 0

        self.selected_items = []

    def closeEvent(self, event):
        """Handles what happens with items when dialog is closed"""

        if self.exit_status != 1:
            self.selected_items = []

        print(self.selected_items)
        print("Data Dialog closed!")

    def update_selected_items(self):
        """Keeps the selected items list updated"""

        items = self.listWidget.selectedItems()
        self.selected_items = [item.text() for item in items]

    def handle_ok_button(self):
        """Handles what happens when Ok button is clicked"""

        self.exit_status = 1
        self.close()

    def handle_cancel_button(self):
        """Handles what happens when Cancel button is clicked"""

        self.exit_status = 0
        print("Canceled!")
        self.close()

    def handle_checkbox(self):
        """Handles what happens when checkbox is toggled"""

        self.choose_colors = not self.choose_colors

class ColorsDialog(QtWidgets.QDialog):
    """A dialog for selecting behavior colors"""

    def __init__(self, behaviors, parent=None):
        super(ColorsDialog, self).__init__(parent)

        self.layout = QtWidgets.QVBoxLayout()

        for behavior in behaviors:
            behavior_button = QtWidgets.QPushButton(behavior)
            behavior_button.clicked.connect(self.handle_behavior_button)
            # behavior_button.setDefault(False)
            behavior_button.setAutoDefault(False)

            self.layout.addWidget(behavior_button)

        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.clicked.connect(self.on_ok)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)

        self.button_box = QtWidgets.QHBoxLayout()
        self.button_box.addWidget(self.ok_button)
        self.button_box.addStretch()
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)
        self.setLayout(self.layout)

        # Type of exit, (i.e. close, ok, cancel)
        # close: 0; cancel: 0; ok: 1
        self.exit_status = 0

        self.behavior_colors = {}

    def closeEvent(self, event):
        """Handles what happens with items when dialog is closed"""

        if self.exit_status != 1:
            self.behavior_colors = {}

        print("Colors Dialog closed!")

    def handle_behavior_button(self):
        """Displays a dialog for selecting colour values"""

        # Get the name of the button (i.e. the behavior) that was clicked
        button_name = self.sender().text()

        # Get the name of the button (i.e. the behavior) that was clicked
        button_name = self.sender().text()

        color_dialog = QtWidgets.QColorDialog(self)

        # Let users choose a color with a level of transparency
        color_dialog.setOption(QtWidgets.QColorDialog.ShowAlphaChannel)
        options = color_dialog.options()
        color = color_dialog.getColor(options=options)

        self.behavior_colors[button_name] = color.getRgb()

    def on_ok(self):
        """Handles what happens when Ok button is clicked"""

        self.exit_status = 1
        self.close()

    def on_cancel(self):
        """Handles what happens when Cancel button is clicked"""

        self.exit_status = 0
        print("Canceled!")
        self.close()

class AxisDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super(AxisDialog, self).__init__(parent)

        self.vert_layout = QtWidgets.QVBoxLayout()

        self.hbox = QtWidgets.QHBoxLayout()

        lower_axis_label = QtWidgets.QLabel("Lower bound:")
        self.lower_axis = QtWidgets.QLineEdit()
        self.lower_axis.setValidator(QtGui.QIntValidator())

        upper_axis_label = QtWidgets.QLabel("Upper bound:")
        self.upper_axis = QtWidgets.QLineEdit()
        self.upper_axis.setValidator(QtGui.QIntValidator())

        self.hbox.addWidget(lower_axis_label)
        self.hbox.addWidget(self.lower_axis)
        self.hbox.addWidget(upper_axis_label)
        self.hbox.addWidget(self.upper_axis)

        self.vert_layout.addLayout(self.hbox)

        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.clicked.connect(self.on_ok)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)

        self.button_box = QtWidgets.QHBoxLayout()
        self.button_box.addWidget(self.ok_button)
        self.button_box.addWidget(self.cancel_button)

        self.vert_layout.addLayout(self.button_box)
        self.setLayout(self.vert_layout)

        self.exit_status = 0

        self.lower_bound = None
        self.upper_bound = None

    def closeEvent(self, event):
        """Handles what happens with items when dialog is closed"""

        if self.exit_status != 1:
            self.lower_bound = None
            self.upper_bound = None
        else:
            self.lower_bound = self.lower_axis.text()
            self.upper_bound = self.upper_axis.text()

        print("lower: {}, upper: {}".format(self.lower_bound, self.upper_bound))

    def on_ok(self):
        """Handles what happens when Ok button is clicked"""

        self.exit_status = 1
        self.close()

    def on_cancel(self):
        """Handles what happens when Cancel button is clicked"""

        self.exit_status = 0
        print("Canceled!")
        self.close()

def main():
    """This is just for testing that the DataDialog class works"""

    app = QtWidgets.QApplication(sys.argv)
    # data_dialog = DataDialog([str(i) for i in range(4)], checkbox=True)
    # data_dialog.show()

    # axis_dialog = AxisDialog()
    # axis_dialog.show()

    color_dialog = ColorsDialog([str(i) for i in range(4)])
    color_dialog.show()
    app.exec_()
    # print(data_dialog.selected_items)

if __name__ == "__main__":
    main()
