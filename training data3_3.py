import pandas as pd
from math import ceil
import numpy as np
from math import floor
#scaled = np.int16(final_data/np.max(np.abs(final_data)) * 32767)
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank, ssc
from scipy import linalg
from scipy.fftpack import fft
import librosa
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sampledatageneration import samples,new_sample,new_samples,samples2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from skopt.plots import plot_convergence
from skopt import gp_minimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


t = 3
sample_rate = 8000

                   
#################################
load0 = librosa.core.load('0.wav',sr=sample_rate)
load1 = librosa.core.load('1.wav',sr=sample_rate)
load2 = librosa.core.load('2.wav',sr=sample_rate)
load3 = librosa.core.load('3.wav',sr=sample_rate)
load4 = librosa.core.load('4.wav',sr=sample_rate)
load5 = librosa.core.load('5.wav',sr=sample_rate)
# =============================================================================
load6 = librosa.core.load('6.wav',sr=sample_rate)
load7 = librosa.core.load('7.wav',sr=sample_rate)
load8 = librosa.core.load('8.wav',sr=sample_rate)
load9 = librosa.core.load('9.wav',sr=sample_rate)
load10 = librosa.core.load('10.wav',sr=sample_rate)
# =============================================================================
load11 = librosa.core.load('11.wav',sr=sample_rate)
load12 = librosa.core.load('12.wav',sr=sample_rate)
load13 = librosa.core.load('13.wav',sr=sample_rate)
load14 = librosa.core.load('14.wav',sr=sample_rate)
load15 = librosa.core.load('15.wav',sr=sample_rate)
# =============================================================================
load16 = librosa.core.load('16.wav',sr=sample_rate)
load17 = librosa.core.load('17.wav',sr=sample_rate)
load18 = librosa.core.load('18.wav',sr=sample_rate)
load19 = librosa.core.load('19.wav',sr=sample_rate)
load20 = librosa.core.load('20.wav',sr=sample_rate)
# =============================================================================
load21 = librosa.core.load('21.wav',sr=sample_rate)
load22 = librosa.core.load('22.wav',sr=sample_rate)
load23 = librosa.core.load('23.wav',sr=sample_rate)
load24 = librosa.core.load('24.wav',sr=sample_rate)
load25 = librosa.core.load('25.wav',sr=sample_rate)
# =============================================================================
load26 = librosa.core.load('26.wav',sr=sample_rate)
load27 = librosa.core.load('27.wav',sr=sample_rate)
load28 = librosa.core.load('28.wav',sr=sample_rate)
load29 = librosa.core.load('29.wav',sr=sample_rate)
load30 = librosa.core.load('30.wav',sr=sample_rate)
# =============================================================================
load31 = librosa.core.load('31.wav',sr=sample_rate)
load32 = librosa.core.load('32.wav',sr=sample_rate)
load33 = librosa.core.load('33.wav',sr=sample_rate)
load34 = librosa.core.load('34.wav',sr=sample_rate)
load35 = librosa.core.load('35.wav',sr=sample_rate)
load36 = librosa.core.load('36.wav',sr=sample_rate)
load37 = librosa.core.load('37.wav',sr=sample_rate)
load38 = librosa.core.load('38.wav',sr=sample_rate)
load39 = librosa.core.load('39.wav',sr=sample_rate)
load40 = librosa.core.load('40.wav',sr=sample_rate)
# =============================================================================
load41 = librosa.core.load('41.wav',sr=sample_rate)
load42 = librosa.core.load('42.wav',sr=sample_rate)
load43 = librosa.core.load('43.wav',sr=sample_rate)
load44 = librosa.core.load('44.wav',sr=sample_rate)
load45 = librosa.core.load('45.wav',sr=sample_rate)
# =============================================================================
# =============================================================================
load46 = librosa.core.load('46.wav',sr=sample_rate)
load47 = librosa.core.load('47.wav',sr=sample_rate)
load48 = librosa.core.load('48.wav',sr=sample_rate)
load49 = librosa.core.load('49.wav',sr=sample_rate)
load50 = librosa.core.load('50.wav',sr=sample_rate)
# =============================================================================
# =============================================================================
load51 = librosa.core.load('51.wav',sr=sample_rate)
load52 = librosa.core.load('52.wav',sr=sample_rate)
load53 = librosa.core.load('53.wav',sr=sample_rate)
load54 = librosa.core.load('54.wav',sr=sample_rate)
load55 = librosa.core.load('55.wav',sr=sample_rate)
load56 = librosa.core.load('56.wav',sr=sample_rate)
load57 = librosa.core.load('57.wav',sr=sample_rate)
load58 = librosa.core.load('58.wav',sr=sample_rate)
load59 = librosa.core.load('59.wav',sr=sample_rate)
load60 = librosa.core.load('60.wav',sr=sample_rate)
load61 = librosa.core.load('61.wav',sr=sample_rate)
load62=  librosa.core.load('62.wav',sr=sample_rate)
load63 = librosa.core.load('63.wav',sr=sample_rate)
load64 = librosa.core.load('64.wav',sr=sample_rate)
load65 = librosa.core.load('65.wav',sr=sample_rate)
load66 = librosa.core.load('66.wav',sr=sample_rate)
load67 = librosa.core.load('67.wav',sr=sample_rate)
load68 = librosa.core.load('68.wav',sr=sample_rate)
load69 = librosa.core.load('69.wav',sr=sample_rate)
load70 = librosa.core.load('70.wav',sr=sample_rate)
load71 = librosa.core.load('71.wav',sr=sample_rate)
load72=  librosa.core.load('72.wav',sr=sample_rate)
load73 = librosa.core.load('73.wav',sr=sample_rate)
load74 = librosa.core.load('74.wav',sr=sample_rate)
load75 = librosa.core.load('75.wav',sr=sample_rate)
load76 = librosa.core.load('76.wav',sr=sample_rate)
load77 = librosa.core.load('77.wav',sr=sample_rate)
load78 = librosa.core.load('78.wav',sr=sample_rate)
load79 = librosa.core.load('79.wav',sr=sample_rate)
load80 = librosa.core.load('80.wav',sr=sample_rate)
load81 = librosa.core.load('81.wav',sr=sample_rate)
load82=  librosa.core.load('82.wav',sr=sample_rate)
load83 = librosa.core.load('83.wav',sr=sample_rate)
load84 = librosa.core.load('84.wav',sr=sample_rate)
load85 = librosa.core.load('85.wav',sr=sample_rate)
load86 = librosa.core.load('86.wav',sr=sample_rate)
load87 = librosa.core.load('87.wav',sr=sample_rate)
load88 = librosa.core.load('88.wav',sr=sample_rate)
load89 = librosa.core.load('89.wav',sr=sample_rate)
load90 = librosa.core.load('90.wav',sr=sample_rate)
load91 = librosa.core.load('91.wav',sr=sample_rate)
load92=  librosa.core.load('92.wav',sr=sample_rate)
load93 = librosa.core.load('93.wav',sr=sample_rate)
load94 = librosa.core.load('94.wav',sr=sample_rate)
load95 = librosa.core.load('95.wav',sr=sample_rate)
load96 = librosa.core.load('96.wav',sr=sample_rate)
load97 = librosa.core.load('97.wav',sr=sample_rate)
load98 = librosa.core.load('98.wav',sr=sample_rate)
load99 = librosa.core.load('99.wav',sr=sample_rate)
load100 = librosa.core.load('100.wav',sr=sample_rate)
# =============================================================================
load101 = librosa.core.load('101.wav',sr=sample_rate)
load102 = librosa.core.load('102.wav',sr=sample_rate)
load103 = librosa.core.load('103.wav',sr=sample_rate)
load104 = librosa.core.load('104.wav',sr=sample_rate)
load105 = librosa.core.load('105.wav',sr=sample_rate)
load106 = librosa.core.load('106.wav',sr=sample_rate)
load107 = librosa.core.load('107.wav',sr=sample_rate)
load108 = librosa.core.load('108.wav',sr=sample_rate)
load109 = librosa.core.load('109.wav',sr=sample_rate)
load110 = librosa.core.load('110.wav',sr=sample_rate)
load111 = librosa.core.load('111.wav',sr=sample_rate)
load112 = librosa.core.load('112.wav',sr=sample_rate)
load113 = librosa.core.load('113.wav',sr=sample_rate)
load114 = librosa.core.load('114.wav',sr=sample_rate)
load115 = librosa.core.load('115.wav',sr=sample_rate)
load116 = librosa.core.load('116.wav',sr=sample_rate)
load117 = librosa.core.load('117.wav',sr=sample_rate)
load118 = librosa.core.load('118.wav',sr=sample_rate)
load119 = librosa.core.load('119.wav',sr=sample_rate)
load120 = librosa.core.load('120.wav',sr=sample_rate)
load121 = librosa.core.load('121.wav',sr=sample_rate)
load122 = librosa.core.load('122.wav',sr=sample_rate)
load123 = librosa.core.load('123.wav',sr=sample_rate)
load124 = librosa.core.load('124.wav',sr=sample_rate)
load125 = librosa.core.load('125.wav',sr=sample_rate)
##########################
load126 = librosa.core.load('126.wav',sr=sample_rate)
load127 = librosa.core.load('127.wav',sr=sample_rate)
load128 = librosa.core.load('128.wav',sr=sample_rate)
load129 = librosa.core.load('129.wav',sr=sample_rate)
load130 = librosa.core.load('130.wav',sr=sample_rate)
load131 = librosa.core.load('131.wav',sr=sample_rate)
load132 = librosa.core.load('132.wav',sr=sample_rate)
load133 = librosa.core.load('133.wav',sr=sample_rate)
load134 = librosa.core.load('134.wav',sr=sample_rate)
load135 = librosa.core.load('135.wav',sr=sample_rate)
load136 = librosa.core.load('136.wav',sr=sample_rate)
load137 = librosa.core.load('137.wav',sr=sample_rate)
load138 = librosa.core.load('138.wav',sr=sample_rate)
load139 = librosa.core.load('139.wav',sr=sample_rate)
load140 = librosa.core.load('140.wav',sr=sample_rate)
load141 = librosa.core.load('141.wav',sr=sample_rate)
load142 = librosa.core.load('142.wav',sr=sample_rate)
load143 = librosa.core.load('143.wav',sr=sample_rate)
load144 = librosa.core.load('144.wav',sr=sample_rate)
load145 = librosa.core.load('145.wav',sr=sample_rate)
load146 = librosa.core.load('146.wav',sr=sample_rate)
load147 = librosa.core.load('147.wav',sr=sample_rate)
load148 = librosa.core.load('148.wav',sr=sample_rate)
load149 = librosa.core.load('149.wav',sr=sample_rate)
load150 = librosa.core.load('150.wav',sr=sample_rate)

fs = sample_rate

#################################

train_load = [load2,load3,load4,load5,load6,load7,load8,load9,load10,load11,
              load12,load13,load14,load15,load16,load17,load18,load19,load20,
              load21,load22,load23,load24,load25,load26,load27,load28,load29,
              load30,load31,load32,load33,load34,load35,load36,load37,load38,
              load39,load40,load41,load42,load43,load44,load45,load46,load47,
              load48,load49,load50,load51,load52,load53,load54,load55,load56,
              load57,load58,load59,load60,load61,load62,load63,load64,load65,
              load66,load67,load68,load69,load70,load71,load72,load73,load74,
              load75,load76,load77,load78,load79,load80,load81,load82,load83,
              load84,load85,load86,load87,load88,load89,load90,load91,load92,
              load93,load94,load95,load96,load97,load98,load99,load100,load101,
              load102,load103,load104,load105,load106,load107,load108,load109,load110,load111,
              load112,load113,load114,load115,load116,load117,load118,load119,load120,
              load121,load122,load123,load124,load125,load126,load127,load128,
              load129,load130,load131,load132,load133,load134,load135,load136,
              load137,load138,load139,load140,load141,load142,load143,load144,
              load145,load146,load147,load148,load149,load150]

data0 = load0[0]
data1 = load1[0]
points_data0 = floor(data0.shape[0]/fs/t)
points_data1 = floor(data1.shape[0]/fs/t)
data0 = data0[:points_data0*fs*t]
data1 = data1[:points_data1*fs*t]

mfcc_0 = mfcc(data0,fs,winlen=t,nfft=t*fs,winstep=t)
mfcc_1 = mfcc(data1,fs,winlen=t,nfft=t*fs,winstep=t)
mfcc_feat = np.concatenate((mfcc_0,mfcc_1))

# =============================================================================
# fbank_0 = logfbank(data0,fs,winlen=t,nfft=t*fs,winstep=t)
# fbank_1 = logfbank(data1,fs,winlen=t,nfft=t*fs,winstep=t)
# fbank_feat = np.concatenate((fbank_0,fbank_1))
# =============================================================================

hop = 0.5
sc_feat_0 = ssc(data0,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t) 
sc_feat_1 = ssc(data1,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t)
sc_feat = np.concatenate((sc_feat_0,sc_feat_1))

# =============================================================================
# rms_feat = np.array([])
# points = fs*t
# data_ampl = np.abs(np.fft.fft(data0))
# data_ampl = data_ampl[1:]
# data_energy = data_ampl ** 2
# energy = np.append(data_energy,data_energy[-1])
# energy = energy.reshape((floor(points),-1))
# rms = librosa.feature.rmse(S=energy)
# rms = rms.T
# rms_feat = np.append(rms_feat,rms)
# data_ampl = np.abs(np.fft.fft(data1))
# data_ampl = data_ampl[1:]
# data_energy = data_ampl ** 2
# energy = np.append(data_energy,data_energy[-1])
# energy = energy.reshape((floor(points),-1))
# rms = librosa.feature.rmse(S=energy)
# rms = rms.T
# rms_feat = np.append(rms_feat,rms)
# =============================================================================


for num,i in enumerate(train_load):
    num = num+2
    data = i[0]
    points_data = floor(data.shape[0]/fs/t)
    data = data[:points_data*fs*t]
    mfcc_data = mfcc(data,fs,winlen=t,nfft=t*fs,winstep=t)
    mfcc_new = np.concatenate((mfcc_feat,mfcc_data))
    if mfcc_new.shape[0] == samples(t,num+1).shape[0]:
        mfcc_feat = mfcc_new
    else:
        print(num)
        break
    #########################################################
    sc_data = ssc(data,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t) 
    sc_new = np.concatenate((sc_feat,sc_data))
    if sc_new.shape[0] == samples(t,num+1).shape[0]:
        sc_feat = sc_new
    else:
        print(num)
        break
    ########################################################
# =============================================================================
#     fbank_data = logfbank(data,fs,winlen=t,nfft=t*fs,winstep=t)
#     fbank_new = np.concatenate((fbank_feat,fbank_data))
#     if fbank_new.shape[0] == samples(t,num+1).shape[0]:
#         fbank_feat = fbank_new
#     else:
#         print(num)
#         break
# =============================================================================
    ########################################################
# =============================================================================
#     data_ampl = np.abs(np.fft.fft(data))
#     data_ampl = data_ampl[1:]
#     data_energy = data_ampl ** 2
#     energy = np.append(data_energy,data_energy[-1])
#     energy = energy.reshape((floor(points),-1))
#     rms = librosa.feature.rmse(S=energy)
#     rms = rms.T
#     rms_feat = np.append(rms_feat,rms)
# =============================================================================
    
    
#rms_feat = rms_feat.reshape(-1,1)   
result = np.concatenate((mfcc_feat,sc_feat),axis=1)
result.shape

np.save('result_150_2_3s',result)

y = samples(t,50+1) # here it shows range and range contains 0 so i include +1
y.shape

# =============================================================================

# =============================================================================
# train_load = [load53,load54,load55,load56,load57,load58,load59,load60,
#               load61,load62,load63,load64,load65,load66,load67,load68,
#               load69,load70,load71,load72,load73,load74,load75,load76,
#               load77,load78,load79,load80,load81,load82,load83,load84,
#               load85,load86,load87,load88,load89,load90,load91,load92,
#               load93,load94,load95,load96,load97,load98,load99,load100]
# data0 = load51[0]
# data1 = load52[0]
# points_data0 = floor(data0.shape[0]/fs/t)
# points_data1 = floor(data1.shape[0]/fs/t)
# data0 = data0[:points_data0*fs*t]
# data1 = data1[:points_data1*fs*t]
# 
# mfcc_0 = mfcc(data0,fs,winlen=t,nfft=t*fs,winstep=t)
# mfcc_1 = mfcc(data1,fs,winlen=t,nfft=t*fs,winstep=t)
# mfcc_feat = np.concatenate((mfcc_0,mfcc_1))
# 
# # =============================================================================
# # fbank_0 = logfbank(data0,fs,winlen=t,nfft=t*fs,winstep=t)
# # fbank_1 = logfbank(data1,fs,winlen=t,nfft=t*fs,winstep=t)
# # fbank_feat = np.concatenate((fbank_0,fbank_1))
# # =============================================================================
# 
# hop = 0.5
# sc_feat_0 = ssc(data0,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t) 
# sc_feat_1 = ssc(data1,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t)
# sc_feat = np.concatenate((sc_feat_0,sc_feat_1))
# 
# # =============================================================================
# # rms_feat = np.array([])
# # points = fs*t
# # data_ampl = np.abs(np.fft.fft(data0))
# # data_ampl = data_ampl[1:]
# # data_energy = data_ampl ** 2
# # energy = np.append(data_energy,data_energy[-1])
# # energy = energy.reshape((floor(points),-1))
# # rms = librosa.feature.rmse(S=energy)
# # rms = rms.T
# # rms_feat = np.append(rms_feat,rms)
# # data_ampl = np.abs(np.fft.fft(data1))
# # data_ampl = data_ampl[1:]
# # data_energy = data_ampl ** 2
# # energy = np.append(data_energy,data_energy[-1])
# # energy = energy.reshape((floor(points),-1))
# # rms = librosa.feature.rmse(S=energy)
# # rms = rms.T
# # rms_feat = np.append(rms_feat,rms)
# # =============================================================================
# 
# 
# for num,i in enumerate(train_load):
#     num = num+51+2
#     data = i[0]
#     points_data = floor(data.shape[0]/fs/t)
#     data = data[:points_data*fs*t]
#     mfcc_data = mfcc(data,fs,winlen=t,nfft=t*fs,winstep=t)
#     mfcc_new = np.concatenate((mfcc_feat,mfcc_data))
#     if mfcc_new.shape[0] == samples2(t,51,num+1).shape[0]:
#         mfcc_feat = mfcc_new
#     else:
#         print(num)
#         break
#     #########################################################
#     sc_data = ssc(data,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t) 
#     sc_new = np.concatenate((sc_feat,sc_data))
#     if sc_new.shape[0] == samples2(t,51,num+1).shape[0]:
#         sc_feat = sc_new
#     else:
#         print(num)
#         break
#     ########################################################
# # =============================================================================
# #     fbank_data = logfbank(data,fs,winlen=t,nfft=t*fs,winstep=t)
# #     fbank_new = np.concatenate((fbank_feat,fbank_data))
# #     if fbank_new.shape[0] == samples2(t,51,num+1).shape[0]:
# #         fbank_feat = fbank_new
# #     else:
# #         print(num)
# #         break
# #     ########################################################
# #     data_ampl = np.abs(np.fft.fft(data))
# #     data_ampl = data_ampl[1:]
# #     data_energy = data_ampl ** 2
# #     energy = np.append(data_energy,data_energy[-1])
# #     energy = energy.reshape((floor(points),-1))
# #     rms = librosa.feature.rmse(S=energy)
# #     rms = rms.T
# #     rms_feat = np.append(rms_feat,rms)
# # =============================================================================
#     
#     
# # rms_feat = rms_feat.reshape(-1,1)   
# result = np.concatenate((mfcc_feat,sc_feat),axis=1)
# result.shape
# 
# result0 = np.load('result_50_2.npy')
# 
# new_result = np.concatenate((result0,result))
# new_result.shape
# 
# #np.save('result_100_2',new_result)
# 
# y = samples2(t,0,100+1) # here it shows range and range contains 0 so i include +1
# y.shape
# =============================================================================

train_load = [load128,load129,load130,load131,load132,load133,
              load134,load135,load136,load137,load138,load139,
              load140,load141,load142,load143,load144,load145,
              load146,load147,load148,load149,load150]
data0 = load126[0]
data1 = load127[0]
points_data0 = floor(data0.shape[0]/fs/t)
points_data1 = floor(data1.shape[0]/fs/t)
data0 = data0[:points_data0*fs*t]
data1 = data1[:points_data1*fs*t]

mfcc_0 = mfcc(data0,fs,winlen=t,nfft=t*fs,winstep=t)
mfcc_1 = mfcc(data1,fs,winlen=t,nfft=t*fs,winstep=t)
mfcc_feat = np.concatenate((mfcc_0,mfcc_1))

# =============================================================================
# fbank_0 = logfbank(data0,fs,winlen=t,nfft=t*fs,winstep=t)
# fbank_1 = logfbank(data1,fs,winlen=t,nfft=t*fs,winstep=t)
# fbank_feat = np.concatenate((fbank_0,fbank_1))
# =============================================================================

hop = 0.5
sc_feat_0 = ssc(data0,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t) 
sc_feat_1 = ssc(data1,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t)
sc_feat = np.concatenate((sc_feat_0,sc_feat_1))

# =============================================================================
# rms_feat = np.array([])
# points = fs*t
# data_ampl = np.abs(np.fft.fft(data0))
# data_ampl = data_ampl[1:]
# data_energy = data_ampl ** 2
# energy = np.append(data_energy,data_energy[-1])
# energy = energy.reshape((floor(points),-1))
# rms = librosa.feature.rmse(S=energy)
# rms = rms.T
# rms_feat = np.append(rms_feat,rms)
# data_ampl = np.abs(np.fft.fft(data1))
# data_ampl = data_ampl[1:]
# data_energy = data_ampl ** 2
# energy = np.append(data_energy,data_energy[-1])
# energy = energy.reshape((floor(points),-1))
# rms = librosa.feature.rmse(S=energy)
# rms = rms.T
# rms_feat = np.append(rms_feat,rms)
# =============================================================================


for num,i in enumerate(train_load):
    num = num+126+2
    data = i[0]
    points_data = floor(data.shape[0]/fs/t)
    data = data[:points_data*fs*t]
    mfcc_data = mfcc(data,fs,winlen=t,nfft=t*fs,winstep=t)
    mfcc_new = np.concatenate((mfcc_feat,mfcc_data))
    if mfcc_new.shape[0] == samples2(t,126,num+1).shape[0]:
        mfcc_feat = mfcc_new
    else:
        print(num)
        break
    #########################################################
    sc_data = ssc(data,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t) 
    sc_new = np.concatenate((sc_feat,sc_data))
    if sc_new.shape[0] == samples2(t,126,num+1).shape[0]:
        sc_feat = sc_new
    else:
        print(num)
        break
    ########################################################
# =============================================================================
#     fbank_data = logfbank(data,fs,winlen=t,nfft=t*fs,winstep=t)
#     fbank_new = np.concatenate((fbank_feat,fbank_data))
#     if fbank_new.shape[0] == samples2(t,51,num+1).shape[0]:
#         fbank_feat = fbank_new
#     else:
#         print(num)
#         break
#     ########################################################
#     data_ampl = np.abs(np.fft.fft(data))
#     data_ampl = data_ampl[1:]
#     data_energy = data_ampl ** 2
#     energy = np.append(data_energy,data_energy[-1])
#     energy = energy.reshape((floor(points),-1))
#     rms = librosa.feature.rmse(S=energy)
#     rms = rms.T
#     rms_feat = np.append(rms_feat,rms)
# =============================================================================
    
    
# rms_feat = rms_feat.reshape(-1,1)   
result = np.concatenate((mfcc_feat,sc_feat),axis=1)
result.shape

result0 = np.load('result_125_2.npy')

new_result = np.concatenate((result0,result))
new_result.shape

#np.save('result_150_2',new_result)

y = samples2(t,0,150+1) # here it shows range and range contains 0 so i include +1
y.shape

idx = np.random.permutation(len(new_result))
x,y = new_result[idx], y[idx]





## Testing data set
loada55 = librosa.core.load('a55.wav',sr=sample_rate)                      
data30 = loada55[0]
points_data30 = floor(data30.shape[0]/fs/t)
data30 = data30[:points_data30*fs*t]

mfcc_in = mfcc(data30,fs,winlen=t,nfft=t*fs,winstep=t) #my frame length is t*fs with 0 padding

sc_in = ssc(data30,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t) #my frame length is t*fs with 0 padding

# fbank_in = logfbank(data30,fs,winlen=t,nfft=t*fs,winstep=t)

# =============================================================================
# points_t = fs*t
# data_ampl_t = np.abs(np.fft.fft(data30))
# data_ampl_t = data_ampl_t[1:]
# data_energy_t = data_ampl_t ** 2
# energy_t = np.append(data_energy_t,data_energy_t[-1])
# energy_t.shape
# energy_t = energy_t.reshape((floor(points_t),-1))
# energy_t.shape
# rms_t = librosa.feature.rmse(S=energy_t)
# rms_in = rms_t.T
# =============================================================================

result_test = np.concatenate((mfcc_in,sc_in),axis=1)
result_test.shape

y_test = new_sample(t,'a55') # it shows audio no here it is 5
y_test.shape


labels = clf.predict(result_test)
conf_mat = metrics.confusion_matrix(y_test,labels)

##### 55,67

# =============================================================================

################################################

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap':[False],
    'max_depth': [94],
    'max_features': [5],
    'min_samples_split': [9],
    'min_samples_leaf': [3],
    'n_estimators': [200]
}
# Create a based model
rf = RandomForestClassifier(n_jobs=-1)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv =3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(x, y)
grid_search.best_params_
grid_search.best_score_
# Fit the grid search to the data
best_grid = grid_search.best_estimator_
best_grid.score(x, y) 
best_grid.score(result_test,y_test)

labels = best_grid.predict(result_test)
conf_mat = metrics.confusion_matrix(y_test,labels)

from sklearn.model_selection import cross_val_score 
cross_val_score(best_grid, result, y, cv=39, n_jobs=-1)

#######################################################

################################################

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'learning_rate':[0.03,0.1,0.3],
    'max_depth': [5,7,10],
    'max_features': [0.04,0.05],
    'min_samples_leaf': [2,3],
    'min_samples_split': [0.001,0.01,0.1],
    'n_estimators': [500,600]
}
# Create a based model
clf = GradientBoostingClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                          cv =50, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(mfcc_feat, y)
grid_search.best_params_
grid_search.best_score_
# Fit the grid search to the data
best_grid = grid_search.best_estimator_
best_grid.score(mfcc_feat, y) 
best_grid.score(mfcc_in,y_test)

from sklearn.model_selection import cross_val_score 
cross_val_score(best_grid, mfcc_feat, y, cv=20, n_jobs=-1)

#######################################################
#

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'C': [0.05,0.001,0.0005],
    'kernel': ['linear']
    
}
# Create a based model
sv = SVC()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = sv, param_grid = param_grid, 
                          cv = 20, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(mfcc_feat, y)
grid_search.best_params_
grid_search.best_score_
# Fit the grid search to the data
best_grid = grid_search.best_estimator_
best_grid.score(mfcc_feat, y) 
best_grid.score(mfcc_in,y_test)

from sklearn.model_selection import cross_val_score 
cross_val_score(best_grid, mfcc_feat, y, cv=20, n_jobs=-1)

# =============================================================================
sv = SVC(kernel = 'linear',C= 0.001)
sv.fit(result,y)
sv.score(result,y)
sv.score(result_test,y_test)


#rf = RandomForestClassifier()
#rf.fit(result,y)    
#rf.score(result,y)
#rf.score(result_test,y_test)

# lr = LogisticRegressionCV()
# lr.fit(result,y)
# lr.score(result,y)
# lr.score(result_test,y_test)
# 
# Specify Gaussian Processes with fixed and optimized hyperparameters
gp_opt = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
gp_opt.fit(result,y)
gp_opt.score(result,y)
gp_opt.score(result_test,y_test)
# =============================================================================

################################################################################

clf = GradientBoostingClassifier(verbose=1)

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
n_features = result.shape[1]

dim_max_depth = Integer(1,35, name='max_depth')
dim_learning_rate = Real(10**-5, 10**0, "log-uniform", name='learning_rate')
dim_max_features = Integer(1, n_features, name='max_features')
dim_min_samples_split = Integer(2, 100, name='min_samples_split')
dim_min_samples_leaf = Integer(1, 100, name='min_samples_leaf')


dimensions = [dim_max_depth,
              dim_learning_rate,
              dim_max_features,
              dim_min_samples_split,
              dim_min_samples_leaf]

default_parameters = [10,10**-2,12,2,1]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(dimensions)
def objective(max_depth,learning_rate,max_features,min_samples_split,min_samples_leaf):
    clf.set_params(max_depth=max_depth,
                   learning_rate=learning_rate,
                   max_features=max_features,
                   min_samples_split=min_samples_split,
                   min_samples_leaf=min_samples_leaf)

    return -np.mean(cross_val_score(clf, x, y, cv=2, n_jobs=-1))

res_gp = gp_minimize(objective,
                     dimensions=dimensions,
                     acq_func='EI', # Expected Improvement.
                     n_calls=60,
                     x0=default_parameters,
                     verbose=True)

res_gp.fun
res_gp.x

space = res_gp.space
sorted(zip(res_gp.func_vals, res_gp.x_iters))

    

plot_convergence(res_gp)



########################


labels = clf.predict(result_test)
conf_mat = metrics.confusion_matrix(y_test,labels)


####################################################################################################
clf = RandomForestClassifier(n_jobs=-1,bootstrap=False,n_estimators=200)

n_features = x.shape[1]

dim_max_depth = Integer(1,300, name='max_depth')
dim_max_features = Integer(1, n_features, name='max_features')
dim_min_samples_split = Integer(2, 50, name='min_samples_split')
dim_min_samples_leaf = Integer(1, 50, name='min_samples_leaf')
dim_criterion = Categorical(['gini','entropy'], name='criterion')

dimensions = [dim_max_depth,
              dim_max_features,
              dim_min_samples_split,
              dim_min_samples_leaf,
              dim_criterion]

default_parameters = [94,5,9,3,'gini']

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(dimensions)
def objective(max_depth,max_features,min_samples_split,min_samples_leaf,criterion):
    clf.set_params(max_depth=max_depth,
                   max_features=max_features,
                   min_samples_split=min_samples_split,
                   min_samples_leaf=min_samples_leaf,
                   criterion=criterion)

    return -np.min(cross_val_score(clf, x, y, cv=3, n_jobs=-1))

res_gp = gp_minimize(objective,
                     dimensions=dimensions,
                     n_calls=80,
                     x0=default_parameters,
                     verbose=True)

res_gp.fun
res_gp.x

space = res_gp.space
sorted(zip(res_gp.func_vals, res_gp.x_iters))

    

plot_convergence(res_gp)


clf = RandomForestClassifier(n_jobs=-1,bootstrap=False,n_estimators=200,max_depth=94,max_features=5,min_samples_split=9,min_samples_leaf=3)
clf.fit(x,y)


####################################################################################################


load_input = librosa.core.load('2.wav',sr=4000)
fs = load_input[1]
data_input = load_input[0]

points_in = floor(data_input.shape[0]/fs/t)
data_input = data_input[:points_in*fs*t]

mfcc_in = mfcc(data2,fs,winlen=t,nfft=t*fs,winstep=t) #my frame length is t*fs with 0 padding

labels = gp_opt.predict(mfcc_in)

########################

labels = pd.Series(labels)
#cluster = pd.Series(labels.value_counts()).idxmax()
clean_samples = labels[labels==2].index
clean_samples


data_in_m = data30.reshape((int(data30.shape[0]/(fs*t))),-1)

clean_samples_m = np.zeros(shape = data_in_m.shape,dtype = 'float32')
for i in clean_samples:
    clean_samples_m[i,:] = data_in_m[i,:]
    
cut = pd.Series(labels[labels!=2]).index
cut

music_free = np.delete(clean_samples_m,cut,axis=0)
music_free.shape

music_free[0,:]

data_in_m[31,:]

final_data = music_free.ravel()

wavfile.write('output2.wav',fs,final_data)





