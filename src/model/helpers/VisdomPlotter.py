from visdom import Visdom


class VisdomPlotter:
    def __init__(self):
        self.viz = Visdom(env='BachPropagation')
        self.plots = {}

    def plot_line(self, plot_name, line_label=None, title=None, y_label=None, x=None, y=None, color=None):
        if plot_name not in self.plots:
            opts = {'title': title, 'xlabel': 'Epochs', 'ylabel': y_label, 'linecolor': color}
            if line_label is not None:
                opts['legend'] = [line_label]
            self.plots[plot_name] = self.viz.line(X=x, Y=y, opts=opts)
        else:
            self.viz.line(X=x, Y=y, win=self.plots[plot_name], name=line_label, update='append',
                          opts={'linecolor': color})

    def display_matplot_figure(self, figure, plot_name):
        if plot_name not in self.plots:
            self.plots[plot_name] = self.viz.matplot(figure)
        else:
            self.viz.matplot(figure, win=self.plots[plot_name])

    def add_song(self, path):
        self.viz.audio(audiofile=path, tensor=None)
