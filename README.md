# Speech-segmentation
Given earning call or any audio containing speech, music, and silence. This code should be able to segment the audio and return clean speech and remove music and silence.

Consider we have input audio of 1 hour which has total of 20 minutes music/silence and 40 minutes speech so my task was to output audio containing no music using machine learning model.
Looking at different audios I have discovered that all earning calls having music at start and at end.
Moreover looking at spectrogram, it is easily discoverable that low frequency range is more important than high frequency range. 

So, instead of analyzing data in 44.1 KHZ, I have chosen frequency to be 8 KHZ which decreases about more than 5 % of data points without any considerable damage to performance.
Now the next step was to extract important features. Features were extracted based on research papers study combined with unsupervised learning. 
Research suggested features like, zero crossing rate,root mean square energy, mel frequency cepstral coefficients, spectral centroid, spectral roll-off etc. 
