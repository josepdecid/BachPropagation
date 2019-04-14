from visdom import Visdom


class VisdomPlotter:
    def __init__(self):
        self.viz = Visdom(env='BachPropagation')
        self.plots = {}

    def plot_line(self, plot, line, title, y_label, x, y, color=None):
        if plot not in self.plots:
            opts = {'title': title, 'xlabel': 'Epochs', 'ylabel': y_label, 'linecolor': color}
            if line is not None:
                opts['legend'] = [line]
            self.plots[plot] = self.viz.line(X=x, Y=y, opts=opts)
        else:
            self.viz.line(X=x, Y=y, win=self.plots[plot], name=line, update='append', opts={'linecolor': color})

    def add_song(self, path):
        self.viz.audio(audiofile=path, tensor=None)