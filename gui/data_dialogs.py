"""
This module contains all the dialogs to present to a user.

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""

import sys

from PyQt5 import QtWidgets, QtCore

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

        # Fill QList with column names that were passed in
        for behavior in behaviors:
            behavior_button = QtWidgets.QPushButton(behavior)
            behavior_button.clicked.connect(self.handle_behavior_button)
            # behavior_button.setDefault(False)
            behavior_button.setAutoDefault(False)
            self.layout.addWidget(behavior_button)

        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.clicked.connect(self.handle_ok_button)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.handle_cancel_button)

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

        color = QtWidgets.QColorDialog.getColor()
        self.behavior_colors[button_name] = color.getRgb()

    def handle_ok_button(self):
        """Handles what happens when Ok button is clicked"""

        self.exit_status = 1
        self.close()

    def handle_cancel_button(self):
        """Handles what happens when Cancel button is clicked"""

        self.exit_status = 0
        print("Canceled!")
        self.close()

def main():
    """This is just for testing that the DataDialog class works"""

    app = QtWidgets.QApplication(sys.argv)
    data_dialog = DataDialog([str(i) for i in range(4)], checkbox=True)
    data_dialog.show()
    app.exec_()
    # print(data_dialog.selected_items)

if __name__ == "__main__":
    main()
