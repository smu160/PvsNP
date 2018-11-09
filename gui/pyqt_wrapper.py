# import pyqtgraph.examples
# pyqtgraph.examples.run()

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class MyWidget(pg.GraphicsWindow):

    def __init__(self, data_q, plots, plot_names, parent=None):
        super().__init__(parent=parent)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10) # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)

        self.vertical_lines = []
        self.plots = plots
        self.add_plots(plot_names)

        self.data_q = data_q

    def onNewData(self):
        val = self.data_q.get()
        
        for v_line in self.vertical_lines:
            v_line.setValue(val)
            QtCore.QCoreApplication.processEvents() 

    def add_plots(self, plot_names):
        pen = pg.mkPen('r', width=2)

        for i in range(len(self.plots)):
            print(self.plots[i])
            plotItem = self.addPlot(title="plot {}".format(plot_names[i]), row=i, col=0)
            plotItem.plot(self.plots[i], pen=pg.mkPen('b', width=2))
            self.vertical_line = plotItem.addLine(x=0, pen=pen)
            self.vertical_lines.append(self.vertical_line)

def main():
    app = QtWidgets.QApplication([])

    pg.setConfigOptions(antialias=True) # True seems to work as well

    win = MyWidget()
    win.show()
    win.resize(800,600)
    win.raise_()
    app.exec_()

if __name__ == "__main__":
    main()
