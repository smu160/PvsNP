import sys
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class MyWidget(pg.GraphicsWindow):

    def __init__(self, data_q, plots, plot_names, beh_intervals=None, parent=None):
        super().__init__(parent=parent)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.start()
        self.timer.timeout.connect(self.on_new_data)

        self.vertical_lines = []
        self.plots = plots
        self.add_plots(plot_names, beh_intervals)
        self.data_q = data_q

    def on_new_data(self):
        try:
            val = self.data_q.get(block=False)
        except:
            return

        for v_line in self.vertical_lines:
            v_line.setValue(val)
            QtCore.QCoreApplication.processEvents()

    def add_plots(self, plot_names, all_beh_intervals):
        x_max = len(self.plots[0])
        pen = pg.mkPen('r', width=2)
        colors = [(0, 0, 255, 50), (255, 165, 0, 50), (255, 0, 0, 50), (0, 255, 0, 50)]
        beh_brushes = [pg.mkBrush(color) for color in colors]

        for i in range(len(self.plots)):
            plot_item = self.addPlot(title="plot {}".format(plot_names[i]), row=i, col=0)
            plot_item.plot(self.plots[i], pen=pg.mkPen('b', width=2))

            # Set the domain and range for each plot
            y_max = self.plots[i].max()
            plot_item.setRange(xRange=[0, x_max], yRange=[0, y_max], padding=0)

            # Add background color(s) (color coded by behavior) to each plot
            if all_beh_intervals:
                for j, behavior_intervals in enumerate(all_beh_intervals):
                    for interval in behavior_intervals:
                        rgn = pg.LinearRegionItem(values=[interval[0], interval[-1]], movable=False)
                        rgn.lines[0].setPen((255, 255, 255, 5))
                        rgn.lines[1].setPen((255, 255, 255, 5))
                        rgn.setBrush(beh_brushes[j])
                        plot_item.addItem(rgn)

            # Create and add vertical line that scrolls
            self.vertical_line = plot_item.addLine(x=0, pen=pen)
            self.vertical_lines.append(self.vertical_line)


def main():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)

    win = MyWidget()
    win.show()
    win.resize(800, 600)
    win.raise_()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
