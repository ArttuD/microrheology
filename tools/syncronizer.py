from glob import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import json
from scipy import interpolate,signal
from matplotlib.widgets import Slider,Button

class syncronizer:

    
    def __init__(self,master_path,used,auto_on,avg_preprocess):
        self.master_path = master_path
        self.used = used
        self.auto_on = auto_on
        self.avg_preprocess = avg_preprocess
        self.large_val = 0.0
        self.small_val = 0.0
    
    
    def syncronizer(self):  
        #Find timesamps: current, track, track_data
        search_path = '/'.join([self.master_path,'*','*_trial_*'])
        search_path_frame = '/'.join([self.master_path,'*','frame_info_matlab.txt'])
        search_path_track = '/'.join([self.master_path,'*','track.json'])
        print(search_path)
        # check if top or local level folder
        if len(glob(search_path)) == 0:
            search_path = '/'.join([self.master_path,'*_trial_*'])
            search_path_frame = '/'.join([self.master_path,'frame_info_matlab.txt'])
            search_path_track = '/'.join([self.master_path,'track.json'])

        print('Search string: {}'.format(search_path))
        files = glob(search_path)
        files_frame = glob(search_path_frame)
        files_track = glob(search_path_track)

        #calculate global displacement, estimate shift, and display
        count = 0
        for trial,info,track in tqdm(list(zip(files,files_frame,files_track))):
            with open(track,'r') as f:
                data = json.load(f)

            fig,ax = plt.subplots(1,1,figsize=(15,8))

            raw = pd.read_csv(trial,sep='\t',header=None)
            timestamps = pd.read_csv(info,sep=' ',index_col=0,header=None)
            stamps = timestamps[1].values
            stamps_2 = raw[8].values
                
            x = np.array([data[i]['x'] for i in list(data.keys())])
            m = np.mean(x,axis=0)
            m = np.gradient(m,2)
            ab = np.abs(m)
            per = np.percentile(ab,99)
            m[(ab<per)] = 0
            m = np.abs(m)
            f = interpolate.interp1d(stamps, m,fill_value="extrapolate")
            ynew = f(stamps_2)

            m2 = np.gradient(raw[0].values,2)
            m2 = np.abs(m2)
            m2[m2<0.05] = 0


            #Estimating the shift
            corr = signal.correlate(ynew, m2)
            lags = signal.correlation_lags(len(m2), len(ynew))
            corr /= np.max(corr)
            m_i = np.argmax(corr)

            
            y_data = signal.detrend(np.mean(x,axis=0)-np.mean(x))
            y_data /= np.std(y_data)
            self.y_data = y_data
            automatic = round(lags[m_i]*0.001,2)
            line, = ax.plot(stamps-automatic,self.y_data)
            ax.plot(stamps_2,raw[0].values)
            name = os.path.split(info)[0]
            name1 = os.path.split(name)[1]
            ax.set_title('shift: {}'.format(automatic))
            fig.suptitle('{}'.format(name1))

            #Manual coarse and fine tuning of shift, and y-axis
            axfreq = plt.axes([0.15, 0.05, 0.65, 0.03])
            freq_slider = Slider(
                ax=axfreq,
                label='Shift (s)',
                valmin=-40,
                valmax=40,
                valstep=0.1,
                valinit=0.0,
            )
            axfreq2 = plt.axes([0.15, 0.01, 0.65, 0.03])
            freq_slider2 = Slider(
                ax=axfreq2,
                label='Shift (s) subscale',
                valmin=-0.2,
                valmax=0.2,
                valinit=0.0,
            )
            
            axamp = plt.axes([0.05, 0.25, 0.0225, 0.63])
            lim = 2*np.std(self.y_data)
            amp_slider = Slider(
                ax=axamp,
                label="Amplitude",
                valmin=-lim,
                valmax=lim,
                valinit=0,
                orientation="vertical"
            )
            
            #single particle and zerp buttons
            axbutton = plt.axes([0.9, 0.02, 0.05, 0.075])
            button = Button(axbutton, 'Local',color='green')

            axbutton2 = plt.axes([0.9, 0.1, 0.05, 0.075])
            button2 = Button(axbutton2, 'average',color='gray')

            def enable(val):
                self.auto_on
                self.auto_on = not self.auto_on
                if self.auto_on:
                    button.color= 'green'
                else:
                    button.color = 'red'
            button.on_clicked(enable)

            def preprocess(val):
                self.avg_preprocess
                self.avg_preprocess = not self.avg_preprocess
                if self.avg_preprocess:
                    button2.label.set_text('average')
                    self.y_data = signal.detrend(np.mean(x,axis=0)-np.mean(x))
                else:
                    button2.label.set_text('big boi')
                    x_ = np.array([data[i]['x'] for i in list(data.keys()) if 'big_' in i])[0]
                    x_norm = (x_-np.mean(x_))/(np.std(x_))
                    self.y_data = signal.detrend(x_norm)

                line.set_ydata(self.y_data+amp_slider.val)
            button2.on_clicked(preprocess)

            def update(val):
                self.used = True
                self.large_val = freq_slider.val
                self.small_val = freq_slider2.val
                extra = 0.0
                if self.auto_on:
                    extra = automatic
                line.set_xdata(stamps-freq_slider.val-freq_slider2.val-extra)
                fig.canvas.draw_idle()
            
            def update_y(val):
                line.set_ydata(self.y_data+amp_slider.val)
                fig.canvas.draw_idle()

            #update 
            freq_slider.on_changed(update)
            freq_slider2.on_changed(update)
            amp_slider.on_changed(update_y)
            fig.show()
            plt.show()
            out_path = os.path.join(os.path.split(info)[0],'sync.npy')
            val = lags[m_i]*0.001
            if self.used:
                val = self.large_val+self.small_val
                if self.auto_on:
                    val += automatic
                print("Using manual({}): {}".format('A' if self.auto_on else 'M',val))
            else:
                print("Using automatic: {}".format(val))
            np.save(out_path,val)
