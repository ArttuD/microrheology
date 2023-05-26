import os
import numpy as np
class ManualChoice:
    def __init__(self,figure,axes,mask,out_path,save=False):
        self.figure = figure
        if isinstance(axes,np.ndarray):
            self.axes = axes.ravel()
        else:
            self.axes = [axes]
        self.out_path = out_path
        self.save = save
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self)
        self.close = self.figure.canvas.mpl_connect('close_event', self.on_close)

        self.default_mask = mask.copy()
        self.mask = mask.copy()

    def find(self,ax):
        for idx,i in enumerate(self.axes):
            if i == ax:
                return idx
        return 0

    def on_close(self,event):
        if self.save:
            self.figure.savefig(self.out_path)


    def __call__(self, event):
        if event.inaxes in self.axes:
            a_id = self.find(event.inaxes)
            # this is a labeled as ok
            if a_id<len(self.default_mask) and self.default_mask[a_id]:
                self.mask[a_id] = not self.mask[a_id] 
                col = (1.,1.,1.) if self.mask[a_id] else (212/255, 0., 123/255)
                self.axes[a_id].set_facecolor(col)
                self.axes[a_id].figure.canvas.draw()

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self.cid)
        self.figure.canvas.mpl_disconnect(self.close)