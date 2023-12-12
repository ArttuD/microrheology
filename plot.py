import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import argparse
import os
import sys
from glob import glob
from tools.util import find_plot_size
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import configparser
import logging
from tools.helpers.ManualChoice import ManualChoice



config = configparser.ConfigParser()
config_path = sys.argv[1]
# read specified config file?
if config_path.endswith('.ini'):
    c_path = os.path.join(os.getcwd(),'configs',config_path)
    if os.path.exists(c_path):
        config.read(c_path)
    else:
        raise NotADirectoryError(c_path)
else:
    # otherwise get the latest
    config.read(os.path.join(os.getcwd(),'configs','default.ini'))

partice_size_choises = list(config['BEADS'].keys())
zoom_choices = list(config['PIXELS'].keys())

parser = argparse.ArgumentParser(
    description="""Download results in the folder and ouputs results
                """)
parser.add_argument('--path','-p',required=True,
                    help='Path to folder. eg. C:/data/imgs')
parser.add_argument('--label','-l',default='holder',
                    help='Give x-label for the plots')
parser.add_argument('--particle_size','-s',required=True,
                    choices=partice_size_choises,
                    help='Give the size of the particles.')
parser.add_argument('--pixel_size',required=True,
                    choices=zoom_choices, 
                    help='Determine pixel size.')
parser.add_argument('--flip','-f',help='Flip sinusoids',
                    action="store_true")
parser.add_argument('--manual','-m',help='Manually remove poor tracks',
                    action="store_true")
parser.add_argument('-d', '--debug',
                    help="Print lots of debugging statements",
                    action="store_const", dest="loglevel",
                    const=logging.DEBUG,
                    default=logging.WARNING)
parser.add_argument('-v', '--verbose',help="Be verbose",
                    action="store_const", dest="loglevel",
                    const=logging.INFO)


#Save arguments
args = parser.parse_known_args()[0]
path = args.path
x_label =  args.label
zoom = args.pixel_size
particle_size =  args.particle_size
flip = 1. if args.flip else -1.
manual = args.manual

if manual:
    # clear file
    with open(os.path.join(path,'manually_dropped.csv'),'w') as drop:
        pass

m = float(config['PIXELS'][zoom])
F_V = float(config['BEADS'][particle_size])

logging_level = args.loglevel
auto_sync = True

logging.basicConfig(
                format='%(levelname)s:%(message)s',
                level=logging_level)

logging.info(f'F_V: {F_V} pixel to um: {m}')
logging.info(f'Label: {x_label}')

#search files
for fold_names in tqdm(glob('{}/2*'.format(path))):
    # find file names
    path = fold_names
    head_tail = os.path.split(path)
    head_tail2 = os.path.split(head_tail[0])
    
    #Tracking data
    track_path = os.path.join(path,'track_matched.json')
    if not os.path.exists(track_path):
        track_path = os.path.join(path,'track.json')
    logging.info(os.path.join(fold_names,'*'))
    #Current file
    current_path = [i for i in glob(os.path.join(fold_names,'*')) if '_trial_' in i][0]
    frame_info_path = os.path.join(path,'frame_info_matlab.txt')
    
    #Saving path
    results_info_path = os.path.join(path,'results.csv')

    # find better radius estimate file if such exists
    radius_data = None
    radius_file = os.path.join(path,'radius_estimates.json')
    if os.path.exists(radius_file):
        with open(radius_file,'r') as f:
            radius_data = json.load(f)

    #Check sync file
    if not auto_sync and not os.path.exists(results_info_path):
        logging.info("Skipping path {}".format(path))
        continue
    
    #Read check if old sync information exists
    results_info = None
    if not auto_sync:
        results_info = pd.read_csv(results_info_path,delimiter=',')

    # Download tracking data
    with open(track_path,'r') as f:
        tracking_data = json.load(f)

    # Read current file and frame info
    current = pd.read_csv(current_path,sep='\t',header=None)
    stamps = pd.read_csv(frame_info_path,sep=' ',header=None)

    # create a list containing big and small probe indices in tracking data
    big_probe_indices = [i for i in tracking_data.keys() if tracking_data[i]['label']==1]
    small_probe_indices = [i for i in tracking_data.keys() if tracking_data[i]['label']==0]


    # collect info of big probes and visualize their x coordinates
    big_probe_data = []
    num_data = len(big_probe_indices)
    sub = 0
    
    for i in range(len(big_probe_indices)):

        if stamps.shape[0]<len(tracking_data[big_probe_indices[i]]['timestamps']):
            dif = np.diff(stamps[1].values).mean()
            for i2 in range(len(tracking_data[big_probe_indices[i]]['timestamps'])-stamps.shape[0]):
                time_diff = stamps[1].values[-1]+dif
                logging.info('Appending: {0:.2f} to timestamps'.format(time_diff))
                stamps = stamps.append({'1':time_diff}, ignore_index=True)

        x = np.zeros(stamps.shape[0])
        
        # add x coordinates of their corresponding timestamp locations
        # data is also scaled from pixels to micrometers
        x[tracking_data[big_probe_indices[i]]['timestamps']] = (np.array(tracking_data[big_probe_indices[i]]['x'])*m)
        big_probe_data.append(x)

        
    # plot references
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
            # calculate difference aka displacement
            diff = big_probe_data[idx]-x0
            diff[x==0] = 0
            diff[x0==0] = 0
            disps[k].append(diff)


    #Matching timestamps of traking and current 

    #using the first tracked particle
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
            logging.warning("Sync info not found. Skipping")
            continue
    
    # shifted time indices
    # currently hardcoded based on current sequence
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
    bounds = [[-np.inf,0,-np.inf,-np.inf],[np.inf,np.pi/2,np.inf,np.inf]]
    bounds_d = [[-np.inf,-1/40*(0.05*(2*np.pi)),-np.inf,-np.inf],[np.inf,np.pi/2+1/40*(0.05*(2*np.pi)),np.inf,np.inf]]
    I_guess_c = [1.2,0.01,0.01,0.01]
    I_guess_d = [0.25,0.001,0.01,0.01]

    # find phi0 
    p, pcov = curve_fit(func, t, current_values[c_start:c_end],p0=I_guess_c,bounds=bounds,method='dogbox', maxfev = 1000000)
    phi0 = p[1]

    # function to fit displacement
    def func_disp(x,a,phi,c,d):
        return a*np.sin(2*np.pi*0.05*x-phi-phi0)+c+d*x

    #Save
    o_p = os.path.join(path,'curves')
    if not os.path.exists(o_p):
        os.mkdir(o_p)
    out_path = path
    counter = 0
    out_path = os.path.join(out_path,'results_all.csv')
    f = open(out_path,'w')
    f.write('track_id,reference_id,distance(um),Cov_Sum,a_(um),phi_(rad),c,d,G_abs,radius_(m),rmse,inv.rmse,shift_(s),a_error,phi_error,c_error,d_error,x,y\n')

    # fit and visualize displacements
    for k in disps.keys():
        num_data = len(small_probe_indices)
        plot_ind = find_plot_size(num_data)
        fig_fit,ax_fit = plt.subplots(plot_ind,plot_ind,figsize=(2*plot_ind,2*plot_ind))
        sub = 0
        res = []
        mask = [False]*len(disps[k])
        # loop over all references
        for idx,i in enumerate(disps[k]):
            res.append([])
            # move starting time to 0 for trackign data 
            t2 = stamps.values[:,1][d_start:d_end]
            t2 -= t2[0]
            t2 = t2.astype(np.float32)

            #Fit
            sample = flip*disps[k][idx][d_start:d_end]
            sample = gaussian_filter1d(sample,5)
            
            # find fit
            p2 = None
            pcov2 = None
            success = False
            success_iter = 0
            guess = np.copy(I_guess_d)
            
            # retry maximum 100 times
            while (not success) and (success_iter<100):       
                try:
                    p2, pcov2 = curve_fit(func_disp,t2,sample,p0=guess,bounds=bounds_d,method='dogbox',maxfev = 1000000)
                    success = True
                except:
                    guess = [np.random.uniform(-5,5),np.random.uniform(0,np.pi),np.random.normal(0,1),np.random.normal(0,1)]
                success_iter += 1
            if not success:
                logging.warning("Fit failed after 100 retries. Quitting...")
                sys.exit(0)
            
            #Error estimates    
            error = np.sqrt(np.diag(pcov2))
            cov_sum = np.sum(pcov2)
            rmse = np.sqrt(mean_squared_error(sample, func(t2,*p2)))
            #inverse of rmse 
            inv_rmse = 1/rmse
                        
            #Search radius value
            big_probe_loc = np.where(np.array(big_probe_indices)==f'{k}')[0][0]
            if radius_data is not None:
                rad_key = big_probe_indices[big_probe_loc]
                if rad_key in list(radius_data.keys()):
                    radius_pixels = radius_data[rad_key]
                else:
                    logging.warning("key: {} not found in radius info".format(k))
                    radius_pixels = radius_data[list(radius_data.keys())[0]]
            else:
                radius_pixels = np.median(tracking_data[big_probe_indices[big_probe_loc]]['radius'])
                
            radius = radius_pixels*m*10**-6
            
            #Calculate Stiffness
            abs_G = np.abs(F_V*(4/3*np.pi*(radius)**3)/(3*np.pi*2*radius*p2[0]*10**-6))


            #Visualize
            if plot_ind == 1:
                ax_fit.plot(t2,sample,label='data')
                ax_fit.plot(t2,func_disp(t2,*p2),label='fit')
                ax_fit.set_title(r'Ref %i'%(idx))
            else:
                if idx !=0 and idx%plot_ind == 0:
                    sub += 1
                # limit plot drawing
                if idx<=60:
                    ax_fit[sub,idx%plot_ind].plot(t2,sample,label='data')
                    ax_fit[sub,idx%plot_ind].plot(t2,func_disp(t2,*p2),label='fit')
                    ax_fit[sub,idx%plot_ind].set_title(r'Ref %i'%(idx))
                if idx==0:
                    ax_fit[sub,idx%plot_ind].set_ylabel(r'$\mu m$')

            #Distance between magnetic particle and reference
            x_big = np.array(tracking_data[big_probe_indices[big_probe_loc]]['x'])*m
            y_big = np.array(tracking_data[big_probe_indices[big_probe_loc]]['y'])*m
            x_small = np.array(tracking_data[small_probe_indices[idx]]['x'])*m
            y_small = np.array(tracking_data[small_probe_indices[idx]]['y'])*m
            distance = np.mean(np.sqrt((x_big.mean()-x_small.mean())**2+(y_big.mean()-y_small.mean())**2))

            #Find areas under wave half periods
            sample_pre = sample-func_disp(t2,0,0,p2[2],p2[3])
            cur_loc = 0
            found = True
            locs = [cur_loc]
            dx = np.diff(t2).mean()
            for s_loc in range(2):
                if (locs[-1]+20/2)>t2[-1]:
                    found = False
                    break
                test_point = np.where(t2>=locs[-1]+20/2)[0][0]
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
                        new_prop = t2[test_point+t_i]
                        locs.append(new_prop)
                        break
                    if left_test:
                        append = True
                        new_prop = t2[test_point-t_i]
                        locs.append(new_prop)
                        break
                    right_prev = r_new
                    left_prev = l_new
                    t_i += 1
                    if (test_point+t_i)>=t2.shape[0] or (test_point-t_i)<0:
                        cont = False
                if not append:
                    found = False
                    logging.warning("not found zero crossings")
                    break
            
            #Compare areas to eliminate singular signals
            peaks_close = False
            if found:
                areas = []
                for ee in locs:
                    if plot_ind == 1:
                        ax_fit.axvline(ee,color='red')
                    else:
                        ax_fit[sub,idx%plot_ind].axvline(ee,color='red')
                for e in range(1,3):
                    # find areas
                    l = np.where(t2>=locs[e-1])[0][0]
                    l2 = np.where(t2>=locs[e])[0][0]
                    area = np.trapz(sample_pre[l:l2],t2[l:l2])
                    areas.append(area)
                    
                peaks_close = np.isclose(np.abs(areas[0]),np.abs(areas[1]),rtol=0.3,atol=0)

            # dont save if distance between reference and big probe is too high, and peak areas are not equal
            # also drop cases phi are nonsense
            k_num = k.split('_')[-1]
            if distance<=250 and distance>=50 and p2[0] > 0 and peaks_close:
                res[idx] = [k_num,idx,distance,cov_sum,*p2,abs_G,radius,rmse,inv_rmse,float(shift),*error,x_big[0],y_big[0]]
                mask[idx] = True
                counter += 1
            else:
                if not peaks_close:
                    if plot_ind==1:
                        ax_fit.set_facecolor((255/255, 205/255, 69/255))
                    else:
                        ax_fit[sub,idx%plot_ind].set_facecolor((255/255, 205/255, 69/255))
                else:
                    if plot_ind==1:
                        ax_fit.set_facecolor((252/255, 215/255, 212/255))
                    else:
                        ax_fit[sub,idx%plot_ind].set_facecolor((252/255, 215/255, 212/255))

        fig_fit.suptitle('Track %s'%k_num)
        file_name = os.path.join(path, 'Track_{}.jpg'.format(k_num))
        if manual:
            handler = ManualChoice(fig_fit,ax_fit,mask,file_name,save=True)
            plt.show()
            mask_manual = handler.mask
            handler.disconnect()
            # save manually dropped
            with open(os.path.join(path,'manually_dropped.csv'),'a') as drop:
                for r,drop_m,data_m in zip(res,mask_manual,mask):
                    if drop_m != data_m:
                        drop.write(f'{r[0]},{r[1]}\n')
            mask = mask_manual
        else:
            fig_fit.savefig(file_name)
        # write data to file
        for r,to_write in zip(res,mask):
            if to_write:
                st = ('{},'*len(r))[:-1]
                f.write(st.format(*r))
                f.write('\n')
        
    f.close()

###############################
#Summarize and visualize data##
###############################

#Read just saved results again
path = args.path
result_paths = glob(os.path.join(path,'*','results_all.csv'))
idents = list(map(lambda x: os.path.normpath(x).split(os.sep)[-2][:14],result_paths))
sample_types = list(map(lambda x: os.path.normpath(x).split(os.sep)[-2][15:],result_paths))
res_path_2 = '{}/results'.format(path)


if not os.path.exists(res_path_2):
    os.makedirs(res_path_2)

#Save data into pd dataframe 

#read ID
day = list(map(lambda x: int(x[0:6]),idents))
sample = list(map(lambda x: int(x[6:8]),idents))
holder = list(map(lambda x: int(x[8:10]),idents))
location = list(map(lambda x: int(x[10:12]),idents))
repeat = list(map(lambda x: int(x[12:14]),idents))
sample_type = list(map(lambda x: x[12:14],sample_types))

all_data = []
for idx,p in enumerate(result_paths):
    single = pd.read_csv(p)
    single['day'] = day[idx]
    single['sample'] = sample[idx]
    single['holder'] = holder[idx]
    single['location'] = location[idx]
    single['repeat'] = repeat[idx]
    single['type'] = sample_type[idx]
    all_data.append(single)

data = pd.concat(all_data)
data['phi_(rad)'] = data['phi_(rad)'].astype(np.float64) 
data['G_abs'] = data['G_abs'].astype(np.float64) 

#Previous versions masked data to filter out noise
masked_data = data

#calculate phi in deg and loss tangent
masked_data['phi_(deg)']=np.rad2deg(masked_data['phi_(rad)'].astype(np.float64))
masked_data.loc[:,'tan_phi'] = np.tan(masked_data['phi_(rad)'].astype(np.float64))
masked_data['repeat'] = masked_data['repeat'].values.astype(int)

#group and calculate mean at particle level over references and repeats
mm = masked_data.groupby(['day','sample','holder','location','track_id']).mean(numeric_only=True)
mm = mm.reset_index()

#plot sanity check that data is fien
fig = plt.figure(figsize=(5,5))
sns.scatterplot(x='radius_(m)',y='a_(um)',data=masked_data,hue=x_label)
plt.savefig('{}/r_vs_a.png'.format(res_path_2))
fig = plt.figure(figsize=(5,5))
sns.scatterplot(x='G_abs',y='a_(um)',data=masked_data,hue=x_label)
plt.savefig('{}/G_vs_a.png'.format(res_path_2))
fig = plt.figure(figsize=(5,5))
sns.scatterplot(x='phi_(rad)',y='G_abs',data=masked_data, hue = x_label)
plt.savefig('{}/phi_vs_G.png'.format(res_path_2))

#plot actual values
fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, figsize=(15,5))
ax1.set_title(r'G$^*$ ', fontsize=16)
sns.boxplot(ax = ax1, x=x_label,y='G_abs',data=mm)
sns.swarmplot(ax = ax1, x=x_label,y='G_abs',data=mm, hue = "track_id")
ax1.set_ylabel(r"G$^*$ [Pa]")

ax2.set_title(r'$\phi$', fontsize=16)
sns.boxplot(ax = ax2, x=x_label,y='phi_(deg)',data=mm)
sns.swarmplot(ax = ax2, x=x_label,y='phi_(deg)',data=mm, hue = "track_id")
ax2.set_ylabel(r"$\phi$ $[^\circ]$")

#save
plt.savefig('{}/summary.png'.format(res_path_2))
mm.to_csv('{}/summary_ID_level.csv'.format(res_path_2),index=False)
masked_data.to_csv('{}/summary_ref_level.csv'.format(res_path_2),index=False)