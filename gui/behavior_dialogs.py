import sys

from PyQt5 import QtWidgets, QtCore

class BehaviorDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super(BehaviorDialog, self).__init__(parent)

        self.layout = QtWidgets.QVBoxLayout()
        self.listWidget = QtWidgets.QListWidget()
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget.setGeometry(QtCore.QRect(10, 10, 211, 291))

        for i in range(10):
            item = QtWidgets.QListWidgetItem("Behavior_{}".format(i))
            self.listWidget.addItem(item)

        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.clicked.connect(self.handle_ok_button)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.handle_cancel_button)

        self.listWidget.itemClicked.connect(self.update_selected_items)
        self.layout.addWidget(self.listWidget)
        self.layout.addWidget(self.ok_button)
        self.layout.addWidget(self.cancel_button)
        self.setLayout(self.layout)

        # Type of exit, (i.e. close, ok, cancel)
        # close: 0; cancel: 0; ok: 1
        self.exit_status = 0
        self.selected_items = []

    def closeEvent(self,event):
        if self.exit_status != 1:
            self.selected_items = []

        print("Behavior Dialog closed!")

    def update_selected_items(self):
        items = self.listWidget.selectedItems()
        self.selected_items = [str(self.listWidget.selectedItems()[i].text()) for i, _ in enumerate(items)]

    def handle_ok_button(self):
        self.exit_status = 1
        self.close()

    def handle_cancel_button(self):
        self.selected_items = []
        self.exit_status = 0
        print("Canceled!")
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    behavior_dialog = BehaviorDialog()
    behavior_dialog.show()
    app.exec_()
    print(behavior_dialog.selected_items)
