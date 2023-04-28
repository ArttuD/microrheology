# %% 
#import
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
import cv2
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import find_peaks,detrend
from sklearn.metrics import r2_score
import argparse
import os
import sys
from glob import glob
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from util import plot_probe,find_plot_size
# %%

# pixels to micrometer change depending on the camerate adapter
#µm/pixel 
#for x0.63
m = 3.45/(20*0.63)
#for x1
#m = 3.45/(20*1)

#Init 

#fit_legacy = False #Old setup
auto_sync = True #If syncronizer was used
fat_oil = True #If 30Cst silicon oil was used
save_curves = False #Not related
current_long = False #Current seq magnetization period 

counter = 0
#Search all the folders from the dictionary
for fold_names in tqdm(glob("C:/Users/lehtona6/Experiments/220705/*")):
    
    # find file names
    path = fold_names
    head_tail = os.path.split(path)
    head_tail2 = os.path.split(head_tail[0])
    #Download tracks
    track_path = os.path.join(path,'track_matched.json')
    if not os.path.isfile(track_path):
        track_path = os.path.join(path,'track.json')
    #download currents
    current_path = [i for i in glob(os.path.join(fold_names,'*')) if '_trial_' in i][0]
    #Download video timestamps
    frame_info_path = os.path.join(path,'frame_info_matlab.txt')
    #results path
    results_info_path = os.path.join(path,'results_final.csv')


    # find better radius estimate file if such exists
    radius_data = None
    radius_file = os.path.join(path,'radius_estimates.json')
    if os.path.exists(radius_file):
        with open(radius_file,'r') as f:
            radius_data = json.load(f)
    
    #Check if syncronizer was used
    if not auto_sync and not os.path.exists(results_info_path):
        print("Skipping path {}".format(path))
        continue
    
    results_info = None
    if not auto_sync:
        results_info = pd.read_csv(results_info_path,delimiter=',')

    # Download tracking data
    with open(track_path,'r') as f:
        tracking_data = json.load(f)

    # Read current file and frame info
        current = pd.read_csv(current_path,sep='\t',header=None, decimal=",")
        stamps = pd.read_csv(frame_info_path,sep=' ',header=None)

    # create a list containing big and small probe indices in tracking data
    big_probe_indices = [i for i in tracking_data.keys() if tracking_data[i]['label']==1]
    small_probe_indices = [i for i in tracking_data.keys() if tracking_data[i]['label']==0]


    # collect info of big probes and visualize their x coordinates
    big_probe_data = []
    num_data = len(big_probe_indices)
    sub = 0
    #Process magnetic particles
    for i in range(len(big_probe_indices)):
        
        if stamps.shape[0]<len(tracking_data[big_probe_indices[i]]['timestamps']):
            time_diff = stamps[1].values[-1]+np.nanmean(np.diff(stamps[1].values))
            
            for kk in range(len(tracking_data[big_probe_indices[i]]['timestamps'])-stamps.shape[0]):
                print('Appending: {0:.2f} to timestamps'.format(time_diff))
                stamps = stamps.append({'1':time_diff}, ignore_index=True)

        x = np.zeros(stamps.shape[0])
        # add x coordinates of their corresponding timestamp locations
        # data is also scaled from pixels to micrometers
        x[tracking_data[big_probe_indices[i]]['timestamps']] = (np.array(tracking_data[big_probe_indices[i]]['x'])*m)
        big_probe_data.append(x)

    #Process reference particles
    num_data = len(small_probe_indices)
    sub = 0
    for i in range(len(small_probe_indices)):

        ref = np.zeros(stamps.shape[0])
        ref[tracking_data[small_probe_indices[i]]['timestamps']] = tracking_data[small_probe_indices[i]]['x']
    
    # construct a dictionary containg pairs big_probe_index: [<list of all possible displacements>]
    disps = {}
    for i in big_probe_indices:
        disps['{}'.format(i)] = []

    # append all displacements
    for idx,k in enumerate(disps.keys()):
        num_data = len(small_probe_indices)
        sub = 0
        for j,i in enumerate(small_probe_indices):

            small_probe = tracking_data[i]
            x0 = np.zeros(stamps.shape[0])

            # transform coordinates
            x0[small_probe['timestamps']] = np.array(small_probe['x'])*m
            
            # calculate difference aka displacement in respect to ref
            diff = big_probe_data[idx]-x0
            diff[x==0] = 0
            diff[x0==0] = 0
            disps[k].append(diff)

    # syncronization
    # atm using the first tracked particle
    x = big_probe_data[0]
    # Current timestamps and data
    current_times = current.values[:,-1]
    current_values = current.values[:,0]

    #Starting value for shift
    shift = 0.0
    #Normalize x so that it is easier to visualize
    x_norm = np.copy(x)
    x_norm[x_norm==0] = np.mean(x_norm)
    x_norm -= np.mean(x_norm)
    x_norm = (x_norm-np.min(x_norm))/(np.max(x_norm)-np.min(x_norm))
    current_values_norm = (current_values-np.min(current_values))/(np.max(current_values)-np.min(current_values))

    current_x = None
    data_x = None

    shift = 0
    if not auto_sync:
        shift = float(results_info['shift_(s)'][0]) 
    else:
        shift_path = os.path.join(path,'sync.npy')
        if os.path.exists(shift_path):
            shift = np.load(shift_path)
        else:
            print("Sync info not found. Skipping")
            continue
    
    # shifted time indices
    # currently hardcoded and shift from the syncronization 
    #These are either 10 and 40 or 40 and 70 depending on the current file. 
    c_start = np.where(current.values[:,-1]>15)[0][0]
    c_end = np.where(current.values[:,-1]<45)[0][-1]
    d_start = np.where(stamps.values[:,1]>(15+shift))[0][0]
    d_end = np.where(stamps.values[:,1]<(45+shift))[0][-1]
    if current_long == True:
        c_start = np.where(current.values[:,-1]>40)[0][0]
        c_end = np.where(current.values[:,-1]<70)[0][-1]
        d_start = np.where(stamps.values[:,1]>(40+shift))[0][0]
        d_end = np.where(stamps.values[:,1]<(70+shift))[0][-1]


    # function to fit current
    def func(x,a,phi0,c,d):
        return a*np.sin(2*np.pi*0.05*x-phi0)+c+d*x

    # move starting time index to 0 for current data
    t = current_times[c_start:c_end]
    t -= t[0]
    t = t.astype(np.float32)

    # bounds for curve fitting
    bounds = [[-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.pi/2,np.inf,np.inf]]
    bounds_derivative = [[-np.inf,-1/10*(0.05*(2*np.pi)),-np.inf],[np.inf,np.pi/2+1/10*(0.05*(2*np.pi)),np.inf]]
    I_guess_c = [1.2,0.01,0.01,0.01]
    I_guess_d = [2.0,1.5,0.01]

    # find phi0 (Current delay)
    p, pcov = curve_fit(func, t, current_values[c_start:c_end],p0=I_guess_c,bounds=bounds,method='dogbox', maxfev = 1000000)
    r2 = r2_score(func(t,*p),current_values[c_start:c_end]) # We are not using this to anything: nonlinear function
    
    phi0 = p[1]

    # function to fit displacement
    def func_disp(x,a,phi,d):
        return a*2*np.pi*0.05*np.cos(2*np.pi*0.05*x-phi-phi0)+d

    # create output path
    # if not given add to current path
    out_path = path
    out_path = os.path.join(out_path,'results_final.csv')
    o_p = os.path.join(path,'curves')
    if not os.path.exists(o_p):
        os.mkdir(o_p)
    f = open(out_path,'w')
    # write header
    f.write('track_id,reference_id,distance(um),Cov_Sum,a_(um),phi_(rad),d,F_V_fit,F_V_num,F_fit,F_num,v_fit,v_num,radius_(m),r2,rmse,inv.rmse,shift_(s),a_error,phi_error,c_error,x,y\n')

    f_v_info = []
    f_v_info_num = []
    row_labels = []
    col_labels = []
    # fit and visualize displacements
    for idx2,k in enumerate(disps.keys()):
        num_data = len(small_probe_indices)
        plot_ind = find_plot_size(num_data)
        fig_fit,ax_fit = plt.subplots(plot_ind,plot_ind,figsize=(2*plot_ind,2*plot_ind))
        sub = 0
        f_v_single = []
        f_v_single_num = []
        for idx,i in enumerate(disps[k]):

            if idx2==0:
                col_labels.append(idx)
                # move starting time to 0 for trackign data 
            t2 = stamps.values[:,1][d_start:d_end]
            t2 -= t2[0]
            t2 = t2.astype(np.float32)
            tt = np.copy(t2)#[1:]
            #Fit
            sample = disps[k][idx][d_start:d_end]
            sample_out = np.copy(sample)
            # remove outliers from the slope
            sample = np.gradient(gaussian_filter1d(sample,12))/np.gradient(t2)
            # find fit
            p2 = None
            pcov2 = None
            success = False
            success_iter = 0
            guess = np.copy(I_guess_d)
            # fit and retry maximum 100 times
            while (not success) and (success_iter<100):       
                try:
                    p2, pcov2 = curve_fit(func_disp,tt,sample,p0=guess,bounds=bounds_derivative,method='dogbox', maxfev = 1000000)
                    success = True
                except:
                    guess = [np.random.uniform(0,20),np.random.uniform(0,np.pi/2),np.random.normal(0,10)]
                    print("Fit failed, retrying with: {}".format(guess))
                success_iter += 1
            if not success:
                print("Fit failed after 100 retries. Quitting...")
                sys.exit(0)
            r2 = r2_score(func_disp(tt,*p2),sample) #Not used in analysis; nonlinear
            error = np.sqrt(np.diag(pcov2))
            cov_sum = np.sum(pcov2)
            big_probe_loc = np.where(np.array(big_probe_indices)==f'{k}')[0][0]
            
            #Fetch radius value
            if radius_data is not None:
                rad_key = big_probe_indices[big_probe_loc]
                if rad_key in list(radius_data.keys()):
                    radius_pixels = radius_data[rad_key]
                else:
                    print("key: {} not found in radius info".format(k))
                    radius_pixels = radius_data[list(radius_data.keys())[0]]
            else:
                radius_pixels = np.median(tracking_data[big_probe_indices[big_probe_loc]]['radius'])
            
            print("radius std: {:.4f}".format(np.std(tracking_data[big_probe_indices[big_probe_loc]]['radius'])))
            radius = radius_pixels*m*10**-6
            
            #calculate rmse
            rmse = np.sqrt(mean_squared_error(sample, func_disp(tt,*p2)))
            
            #inverse of rmse 
            inv_rmse = 1/rmse
            if plot_ind == 1:
                ax_fit.plot(tt,sample,label='data')
                ax_fit.plot(tt,func_disp(tt,*p2),label='fit')
                ax_fit.set_title(r'Ref %i, R2: %.5f'%(idx,r2))

            if idx !=0 and idx%plot_ind == 0:
                sub += 1
            if idx<=8:
                ax_fit[sub,idx%plot_ind].plot(tt,np.abs(sample),label='data')
                ax_fit[sub,idx%plot_ind].plot(tt,np.abs(func_disp(tt,*p2)),label='fit')
                ax_fit[sub,idx%plot_ind].set_title(r'Ref %i, R2: %.5f'%(idx,r2))
            if idx==0:
                ax_fit[sub,idx%plot_ind].set_ylabel(r'$\mu m$')
            print('-'*10)
            print('track: {}, ref: {}'.format(k,idx))
            print('r2: {:.3f}'.format(r2))
            print('a: {:.2f},phi: {:.2f},d: {:.2f}'.format(*p2))


            x_big = np.array(tracking_data[big_probe_indices[big_probe_loc]]['x'])*m
            y_big = np.array(tracking_data[big_probe_indices[big_probe_loc]]['y'])*m

            x_small = np.array(tracking_data[small_probe_indices[idx]]['x'])*m
            y_small = np.array(tracking_data[small_probe_indices[idx]]['y'])*m

            distance = np.mean(np.sqrt((x_big.mean()-x_small.mean())**2+(y_big.mean()-y_small.mean())**2))
            
            volume = (4/3*np.pi*radius**3)

            # calculate dynamic viscosity
            # given by manufacturer
            #T_0 = 25
            #r_T_0 = 0.97e3
            #v = 1026.16e-6
            #for 30 Cst particles
            T_0 = 25
            r_T_0 = 0.971e3
            v = 1026.16e-6
            if fat_oil ==True:
                v = 30000e-6

            # our experiments
            T = 23
            a = 9.2e-4
            b = 4.5e-7
            r = r_T_0/(1+a*(T-T_0)+b*(T-T_0)**2)
            nn = v*r

            v_fit = np.mean(np.abs(func_disp(tt,p2[0],p2[1],0)*(1e-6)))
            F_fit = 6*np.pi*nn*radius*v_fit
            f_v_fit = F_fit/volume

            v_num = np.mean(np.abs(sample)*(1e-6))
            F_num = 6*np.pi*nn*radius*v_num
            f_v_num_estimate = F_num/volume

            f_v_single.append('{:e}'.format(f_v_fit))
            f_v_single_num.append('{:e}'.format(f_v_num_estimate))


            sample_pre = sample-func_disp(tt,0,0,p2[2])
            cur_loc = 0
            found = True
            locs = [cur_loc]
            dx = np.diff(tt).mean()
            for s_loc in range(2):
                if (locs[-1]+20/2)>tt[-1]:
                    found = False
                    break
                test_point = np.where(tt>=locs[-1]+20/2)[0][0]
                cont = True
                append = False
                t_i = 0
                right_prev = np.sign(sample_pre[test_point])
                left_prev = np.sign(sample_pre[test_point])
                while cont:
                    # try both directions
                    r_new = np.sign(sample_pre[test_point+t_i])
                    l_new = np.sign(sample_pre[test_point-t_i])
                    # this is true if they are different signs (crossing zero) 
                    right_test = np.diff([right_prev,r_new])!=0
                    left_test = np.diff([l_new,left_prev])!=0
                    if right_test:
                        append = True
                        new_prop = tt[test_point+t_i]
                        locs.append(new_prop)
                        break
                    if left_test:
                        append = True
                        new_prop = tt[test_point-t_i]
                        locs.append(new_prop)
                        break
                    right_prev = r_new
                    left_prev = l_new
                    t_i += 1
                    if (test_point+t_i)>=tt.shape[0] or (test_point-t_i)<0:
                        cont = False
                if not append:
                    #found = False
                    print("not found zero crossings")
                    break
            
            peaks_close = False
            if found:
                areas = []
                for ee in locs:
                    ax_fit[sub,idx%plot_ind].axvline(ee,color='red')
                for e in range(1,3):
                    # find areas
                    l = np.where(t2>=locs[e-1])[0][0]
                    l2 = np.where(t2>=locs[e])[0][0]
                    area = np.trapz(sample_pre[l:l2],t2[l:l2])
                    areas.append(area)
                    #print(area)
                #print(areas)
                #print('{} {}'.format(np.abs(np.abs(areas[0])-np.abs(areas[1])),0.1*np.abs(areas[1])))
                peaks_close = np.isclose(np.abs(areas[0]),np.abs(areas[1]),rtol=0.1,atol=0)
                #print(peaks_close)

            # dont save if distance between reference and big probe is too high
            # also drop cases where a or phi are nonsense
            #peaks_close = True
            #if distance<=250 and distance>=20 and p2[0]!=0 and r2>0.0:
            #if True:
            k_num = k.split('_')[-1]
            if distance<=250 and distance>=20 and p2[0]!=0:
                #print(x_big[0])
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(k_num,idx,distance,cov_sum,*p2,f_v_fit,f_v_num_estimate,F_fit,F_num,v_fit,v_num,radius,r2,rmse,inv_rmse,shift,*error,x_big[0],y_big[0]))
                #Would be nice to have both F (from stokes and F_V calibration value saved),and velocity v
                o_p2 = path
                if save_curves:
                    np.save('{}/curves/t_{:03d}.npy'.format(o_p2,counter),t2)
                    np.save('{}/curves/data_{:03d}.npy'.format(o_p2,counter),sample_pre)
                    np.save('{}/curves/radius_{:03d}.npy'.format(o_p2,counter),radius)
                counter += 1
            else:
                if not peaks_close:
                    ax_fit[sub,idx%plot_ind].set_facecolor((255/255, 205/255, 69/255))
                else:
                    ax_fit[sub,idx%plot_ind].set_facecolor((252/255, 215/255, 212/255))

        fig_fit.suptitle('Track %s'%k_num)
        file_name = os.path.join(path, 'Track_{}.jpg'.format(k_num))
        fig_fit.savefig(file_name)
        row_labels.append(k)
        f_v_info.append(f_v_single)
        f_v_info_num.append(f_v_single_num)
        file_name = os.path.join(path, 'Track_2_{}.jpg'.format(k_num))
    plt.show()
    f.close()

    print("saved to: {}".format(out_path))

# %%
