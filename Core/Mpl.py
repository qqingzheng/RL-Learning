import matplotlib
import matplotlib.pyplot as plt
from IPython import display
class Plt(object):
    def __init__(self,id,title):
        plt.ion()
        self.id = id
        self.title = title
    def update(self,**args):
        pass

class ViewTrend(Plt):
    def __init__(self,id,title,x_label,y_label):
        super(ViewTrend,self).__init__(id,title)
        self.is_ipython = 'inline' in matplotlib.get_backend()
        self.x_label = x_label
        self.y_label = y_label
    def update(self,*arg):
        plt.figure(self.id)
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        for y in arg:
            plt.plot(y)
        plt.pause(1)
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())