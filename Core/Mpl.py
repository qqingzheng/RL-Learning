import matplotlib
import matplotlib.pyplot as plt
import torch
from IPython import display
class Plt(object):
    def __init__(self,id,title):
        plt.ion()
        self.id = id
        self.title = title
    def update(self,**args):
        pass
class ViewIMG(Plt):
    def __init__(self,id,title):
        super(ViewIMG,self).__init__(id,title)
        self.is_ipython = 'inline' in matplotlib.get_backend()
    def update(self,img):
        plt.figure(self.id)
        plt.clf()
        plt.title(self.title)
        plt.imshow(img.squeeze(0).permute(1, 2, 0))
        plt.pause(1)
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
class ViewTrend(Plt):
    def __init__(self,id,title,x_label,y_label):
        super(ViewTrend,self).__init__(id,title)
        self.is_ipython = 'inline' in matplotlib.get_backend()
        self.x_label = x_label
        self.y_label = y_label
        self.save = False
    def savefig(self,*arg):
        self.save = True
        self.update(*arg)
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
        if self.save:
            plt.savefig(self.title)
            plt.close()