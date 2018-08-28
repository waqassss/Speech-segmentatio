# Speech-segmentation
Given earning call or any audio containing speech, music, and silence. This code should be able to segment the audio and return clean speech and remove music and silence.

Consider we have input audio of 1 hour which has total of 20 minutes music/silence and 40 minutes speech so my task was to output audio containing no music using machine learning model.
Looking at different audios I have discovered that all earning calls having music at start and at end.
Moreover looking at spectrogram, it is easily discoverable that low frequency range is more important than high frequency range. 

So, instead of analyzing data in 44.1 KHZ, I have chosen frequency to be 8 KHZ which decreases about more than 5 % of data points without any considerable damage to performance.
Now the next step was to extract important features. Features were extracted based on research papers study combined with unsupervised learning. 
Research suggested features like, zero crossing rate,root mean square energy, mel frequency cepstral coefficients, spectral centroid, spectral roll-off etc. 

After having training logs ready and audio files ready then the next step was to use unsupervised learning to check accuracy using different features. The code for that is following:

Extracting features:

mfcc_in = mfcc(data,fs,winlen=t,nfft=t*fs,winstep=t) #my frame length is t*fs with 0 padding

sc_in = ssc(data,fs,winlen=t,nfft=int((t*fs)/hop), winstep=t) #my frame length is t*fs with 0 padding

fbank_in = logfbank(data,fs,winlen=t,nfft=t*fs,winstep=t)

points_t = fs*t
data_ampl_t = np.abs(np.fft.fft(data))
data_ampl_t = data_ampl_t[1:]
data_energy_t = data_ampl_t ** 2
energy_t = np.append(data_energy_t,data_energy_t[-1])
energy_t.shape
energy_t = energy_t.reshape((floor(points_t),-1))
energy_t.shape
rms_t = librosa.feature.rmse(S=energy_t)
rms_in = rms_t.T


Applying different clustering algorithms:

from sklearn.mixture import BayesianGaussianMixture
gm = BayesianGaussianMixture(2,max_iter=1000) 
gm.fit(mfcc_feat)
labels = gm.predict(mfcc_feat)
y_test = new_sample(t,'0')
conf_mat = metrics.confusion_matrix(y_test,labels)

from sklearn.cluster import MeanShift
ms = MeanShift()
ms.fit(mfcc_feat)
labels = ms.labels_
y_test = new_sample(t,'0')
conf_mat = metrics.confusion_matrix(y_test,labels)


from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(affinity="cosine",linkage='complete')
labels = ac.fit_predict(mfcc_feat)
y_test = new_sample(t,'0')
conf_mat = metrics.confusion_matrix(y_test,labels)


from sklearn.cluster import AffinityPropagation
af = AffinityPropagation()
af.fit(mfcc_feat)
labels = af.labels_
y_test = new_sample(t,'0')
conf_mat = metrics.confusion_matrix(y_test,labels)

After looking at confusion matrices,we have to choose best features. I found mfcc and spectral centroid good features to use.
