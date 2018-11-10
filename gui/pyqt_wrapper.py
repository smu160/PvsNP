# import pyqtgraph.examples
# pyqtgraph.examples.run()

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class MyWidget(pg.GraphicsWindow):

    def __init__(self, data_q, plots, plot_names, beh_intervals=None, parent=None):
        super().__init__(parent=parent)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)

        self.vertical_lines = []
        self.plots = plots
        self.add_plots(plot_names, beh_intervals)
        self.data_q = data_q

    def onNewData(self):
        try:
            val = self.data_q.get(block=False)
        except:
            return

        for v_line in self.vertical_lines:
            v_line.setValue(val)
            QtCore.QCoreApplication.processEvents()

    def add_plots(self, plot_names, all_beh_intervals):
        pen = pg.mkPen('r', width=2)
        colors = [(0, 0, 255, 10), (255, 165, 0, 10), (255, 0, 0, 10), (0, 255, 0, 10)]
        beh_brushes = [pg.mkBrush(color) for color in colors]

        for i in range(len(self.plots)):
            print(self.plots[i])
            plotItem = self.addPlot(title="plot {}".format(plot_names[i]), row=i, col=0)
            plotItem.plot(self.plots[i], pen=pg.mkPen('b', width=2))

            # Add background color(s) (color coded by behavior) to each plot
            if all_beh_intervals:
                for i, behavior_intervals in enumerate(all_beh_intervals):
                    for interval in behavior_intervals:
                        rgn = pg.LinearRegionItem([interval[0], interval[-1]], movable=False)
                        rgn.setBrush(beh_brushes[i])
                        plotItem.addItem(rgn)

            # Create and add vertical line that scrolls
            self.vertical_line = plotItem.addLine(x=0, pen=pen)
            self.vertical_lines.append(self.vertical_line)



def main():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)

    win = MyWidget()
    win.show()
    win.resize(800, 600)
    win.raise_()
    app.exec_()

if __name__ == "__main__":
    main()
