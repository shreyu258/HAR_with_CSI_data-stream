import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pywt
import seaborn as sns
import csv
import glob
import time
import os
import warnings




def extract_csi(raw_folder, labels, save=False, win_len=1000, step=200):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array 
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    for label in labels:
        features_label = extract_csi_by_label(raw_folder, label, save, win_len, step)
        
        features_label.to_csv(str(label)+'_STFT_200msec_test_try.csv', index=False)
    return features_label


def extract_csi_by_label(raw_folder, label, save=False, win_len=1000, step=200):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_foler: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array 
        win_len  :  integer, window length
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label '+str(label))
    data_path_pattern = os.path.join(raw_folder, 'input_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    annot_csv_files = [os.path.basename(fname).replace('input_', 'annotation_') for fname in input_csv_files]
    annot_csv_files = [os.path.join(raw_folder, fname) for fname in annot_csv_files]
    index=0
    features_label=pd.DataFrame()
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        features = merge_csi_label(csi_file, label_file, win_len=1000, step=200)
        features_label = pd.concat([features_label, features], axis=0,ignore_index=True)
        index+=1
        print(index)
        
    return features_label


def merge_csi_label(csifile, labelfile, win_len=1000, step=200):
    """
    Merge CSV files into a Numpy Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 90)
    Args:
        csifile  :  str, csv file containing CSI data
        labelfile:  str, csv fiel with activity label 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    activity_count = []
    activity =[]
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            activity.append(line[0])
    activity = pd.DataFrame(np.array(activity))
    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            line_array = np.array([float(v) for v in line])
            # extract the amplitude only
            line_array = line_array[1:91]
            csi.append(line_array[np.newaxis,...])
    csi = np.concatenate(csi, axis=0)
    csi = np.array(csi)
    
    pca_filtred_data=pca_filtering(csi)
    
    a=DWT_feature_extraction(pca_filtred_data,200,levels=12)
    a=pd.DataFrame(STFT_feature_extraction(pca_filtred_data,Window_size=1000,step=200,bins=25))
    
    a['Activity'] = annonation_time_split(activity,time_ms=200)
    
    return pca_filtred_data


    def STFT_feature_extraction(a,Window_size,step,bins):
    # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
        n_fft=Window_size

        # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
        hop_length=step

        # Specify the window type for FFT/STFT
        window_type ='hann'

        spectrogram_librosa = np.abs(librosa.stft(a, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type)) ** 2

        return np.transpose(spectrogram_librosa[0:bins])

def annonation_time_split (data,time_ms):
    annonation_label_df = pd.DataFrame()
    for i in range (0,len(data),int(time_ms)):
        annonation_label_df = pd.concat([annonation_label_df, data[i:i+time_ms].mode()], ignore_index=True)
    return annonation_label_df


def moving_average(data,window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data,window,'same')



def hampel_filter_forloop(input_series, window_size, n_sigmas=3):
    
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    
    # possibly use np.nanmedian 
    for i in range((window_size),(n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices


def pca_filtering(amp):
    constant_offset = np.empty_like(amp)
    filtered_data = np.empty_like(amp)

    for i in range(1,len(amp[0])):
        constant_offset[:,i] = moving_average(amp[:,i],400)

    filtered_data = amp - constant_offset

    for i in range(1,len(amp[0])):
        filtered_data[:,i] = moving_average(filtered_data[:,i],100)

        #eigen value cal
    con_mat = np.cov(filtered_data.T)
    eig_val,eig_vec = np.linalg.eig(con_mat) 
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]
    pca_data = filtered_data.dot(eig_vec)

    outliers_removed = np.empty_like(pca_data[:,1:6])


    for i in range(1,6):
        res, detected_outliers = hampel_filter_forloop(pca_data[:,i], 10)
        outliers_removed[:,i-1] = res


    a=np.average(outliers_removed,axis=1)
    return a


def DWT_feature_extraction(a,time_ms,levels=5):
    empty_df = pd.DataFrame()
    for j in range(0,5):
        for i in range (0,len(a[:,0]),int(time_ms)):
            if i==0:
                test=np.array(compute_dwt_energy(a[i:i+int(time_ms),j], wavelet='db5', levels=levels))
            else:
                test_1=np.array(compute_dwt_energy(a[i:i+int(time_ms),j], wavelet='db5', levels=levels))
                test= np.append(test,test_1)
       
    
        empty_df = empty_df.append(pd.Series(test), ignore_index=True)
    mean_df=(pd.DataFrame(empty_df.mean())).T
    
    a=(mean_df).values
    DWT_features_200ms=pd.DataFrame()
    for k in range (0,a.shape[1],12):
        DWT_features_200ms = pd.concat([DWT_features_200ms, pd.DataFrame(a[:,k:k+12].ravel()).T], axis=0, ignore_index=True)
    k=len(DWT_features_200ms.columns)
    for i in range(1,k):
        DWT_features_200ms = pd.concat([DWT_features_200ms,(DWT_features_200ms[i] - DWT_features_200ms[i-1]).abs()], axis=1,ignore_index= True) 
    
    
    
    return DWT_features_200ms

def compute_dwt_energy(signal, wavelet='db5', levels=None):
    # Perform the DWT on the signal
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    
    # Compute energy for each level
    energy_per_level = []
    for level in range(1, len(coeffs)):
        level_energy = sum(coef**2 for coef in coeffs[level])
        energy_per_level.append(level_energy)
    
    return energy_per_level





if __name__ == "__main__":
    labels = ('standup','run','walk','fall','sitdown')
    raw_folder='/Users/shreyu/Desktop/proiject/model_3/Dataset/Data/'
    

    warnings.filterwarnings("ignore")
    extract_csi(raw_folder, labels, save=False, win_len=1000, step=200)