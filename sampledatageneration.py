import numpy as np
import pandas as pd
from math import ceil,floor


# Loading a file

def samples(t, n_audio):
    y = np.array([])
    for i in range(n_audio):
        
        a = pd.read_excel('data.xlsx', str(i))
        total_pts = ((a['Total min'].dropna()*60)+(a['Total sec'].dropna()))/t
        points = ceil(total_pts)
        tot = np.zeros(points)
        
        ###################################
        
        start_music = (((a[a['class'] == 'Music']['start min'].dropna()*60)+(a[a['class'] == 'Music']['start sec'].dropna()))/t).values
        start_music = np.round(start_music,0)
        end_music = (((a[a['class'] == 'Music']['end min'].dropna()*60)+(a[a['class'] == 'Music']['end sec'].dropna()))/t).values
        end_music = np.round(end_music,0)
        
        ##################################
        
        start_speech = (((a[a['class'] == 'Speech']['start min'].dropna()*60)+(a[a['class'] == 'Speech']['start sec'].dropna()))/t).values
        start_speech = np.round(start_speech,0)
        end_speech = (((a[a['class'] == 'Speech']['end min'].dropna()*60)+(a[a['class'] == 'Speech']['end sec'].dropna()))/t).values
        end_speech = np.round(end_speech,0)
        
        ##################################
        
        start_silence = (((a[a['class'] == 'Silence']['start min'].dropna()*60)+(a[a['class'] == 'Silence']['start sec'].dropna()))/t).values
        start_silence = np.round(start_silence,0)
        end_silence = (((a[a['class'] == 'Silence']['end min'].dropna()*60)+(a[a['class'] == 'Silence']['end sec'].dropna()))/t).values
        end_silence = np.round(end_silence,0)
        
        ######################################
        
        #Generating music data
        try:
            music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2]),
                                    np.arange(start_music[3],end_music[3]),
                                    np.arange(start_music[4],end_music[4])))
        except (IndexError,ValueError) as e:
            try:
                music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2]),
                                    np.arange(start_music[3],end_music[3])))
            
            except (IndexError,ValueError) as e:
                try:
                    music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2])))
                except (IndexError,ValueError) as e:
                    try:
                        music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            music = np.arange(start_music[0],end_music[0]) 
                        except ValueError:
                            pass
        try:
            music = music.astype('int16')
            for i in music:
                tot[i] = 1
        except NameError:
            pass
        
        
        #######################################
            
        #Generating speech data
        try:
            speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2]),
                                    np.arange(start_speech[3],end_speech[3]),
                                    np.arange(start_speech[4],end_speech[4])))
        except (IndexError,ValueError) as e:
            try:
                speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2]),
                                    np.arange(start_speech[3],end_speech[3])))
            
            except (IndexError,ValueError) as e:
                try:
                    speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2])))
                except (IndexError,ValueError) as e:
                    try:
                        speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            speech = np.arange(start_speech[0],end_speech[0]) 
                        except ValueError:
                            pass
                        
        try:
            speech = speech.astype('int16')
            for i in speech:
                tot[i] = 2
        except NameError:
            pass
        
        ######################################
        
        #Generating silence data
        try:
            silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2]),
                                    np.arange(start_silence[3],end_silence[3]),
                                    np.arange(start_silence[4],end_silence[4])))
        
        except (IndexError,ValueError) as e:
            try:
                silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2]),
                                    np.arange(start_silence[3],end_silence[3])))
            except (IndexError,ValueError) as e:
                try:
                    silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2])))
                except (IndexError,ValueError) as e:
                    try:
                        silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            silence = np.arange(start_silence[0],end_silence[0]) 
                        except ValueError:
                            pass
                        
        try:
            silence = silence.astype('int16')
            for i in silence:
                tot[i] = 3
        except NameError:
            pass
        
        if total_pts[0] == int(total_pts[0]):
            y = np.append(y,tot)
        else:
            tot = tot[:-1]
            y = np.append(y,tot)
    return y


###############################################################################################
    
def samples2(t, start,stop):
    y = np.array([])
    for i in range(start,stop):
        a = pd.read_excel('data.xlsx', str(i))
        total_pts = ((a['Total min'].dropna()*60)+(a['Total sec'].dropna()))/t
        points = ceil(total_pts)
        tot = np.zeros(points)
        
        ###################################
        
        start_music = (((a[a['class'] == 'Music']['start min'].dropna()*60)+(a[a['class'] == 'Music']['start sec'].dropna()))/t).values
        start_music = np.round(start_music,0)
        end_music = (((a[a['class'] == 'Music']['end min'].dropna()*60)+(a[a['class'] == 'Music']['end sec'].dropna()))/t).values
        end_music = np.round(end_music,0)
        
        ##################################
        
        start_speech = (((a[a['class'] == 'Speech']['start min'].dropna()*60)+(a[a['class'] == 'Speech']['start sec'].dropna()))/t).values
        start_speech = np.round(start_speech,0)
        end_speech = (((a[a['class'] == 'Speech']['end min'].dropna()*60)+(a[a['class'] == 'Speech']['end sec'].dropna()))/t).values
        end_speech = np.round(end_speech,0)
        
        ##################################
        
        start_silence = (((a[a['class'] == 'Silence']['start min'].dropna()*60)+(a[a['class'] == 'Silence']['start sec'].dropna()))/t).values
        start_silence = np.round(start_silence,0)
        end_silence = (((a[a['class'] == 'Silence']['end min'].dropna()*60)+(a[a['class'] == 'Silence']['end sec'].dropna()))/t).values
        end_silence = np.round(end_silence,0)
        
        ######################################
        
        #Generating music data
        try:
            music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2]),
                                    np.arange(start_music[3],end_music[3]),
                                    np.arange(start_music[4],end_music[4])))
        except (IndexError,ValueError) as e:
            try:
                music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2]),
                                    np.arange(start_music[3],end_music[3])))
            
            except (IndexError,ValueError) as e:
                try:
                    music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2])))
                except (IndexError,ValueError) as e:
                    try:
                        music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            music = np.arange(start_music[0],end_music[0]) 
                        except ValueError:
                            pass
        try:
            music = music.astype('int16')
            for i in music:
                tot[i] = 1
        except NameError:
            pass
        
        
        #######################################
            
        #Generating speech data
        try:
            speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2]),
                                    np.arange(start_speech[3],end_speech[3]),
                                    np.arange(start_speech[4],end_speech[4])))
        except (IndexError,ValueError) as e:
            try:
                speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2]),
                                    np.arange(start_speech[3],end_speech[3])))
            
            except (IndexError,ValueError) as e:
                try:
                    speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2])))
                except (IndexError,ValueError) as e:
                    try:
                        speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            speech = np.arange(start_speech[0],end_speech[0]) 
                        except ValueError:
                            pass
                        
        try:
            speech = speech.astype('int16')
            for i in speech:
                tot[i] = 2
        except NameError:
            pass
        
        ######################################
        
        #Generating silence data
        try:
            silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2]),
                                    np.arange(start_silence[3],end_silence[3]),
                                    np.arange(start_silence[4],end_silence[4])))
        
        except (IndexError,ValueError) as e:
            try:
                silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2]),
                                    np.arange(start_silence[3],end_silence[3])))
            except (IndexError,ValueError) as e:
                try:
                    silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2])))
                except (IndexError,ValueError) as e:
                    try:
                        silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            silence = np.arange(start_silence[0],end_silence[0]) 
                        except ValueError:
                            pass
                        
        try:
            silence = silence.astype('int16')
            for i in silence:
                tot[i] = 3
        except NameError:
            pass
        
        if total_pts[0] == int(total_pts[0]):
            y = np.append(y,tot)
        else:
            tot = tot[:-1]
            y = np.append(y,tot)
    return y



########################################################
def new_sample(t,audio_no):    
    
    a = pd.read_excel('data.xlsx', str(audio_no))
    total_pts = ((a['Total min'].dropna()*60)+(a['Total sec'].dropna()))/t
    points = ceil(total_pts)
    tot = np.zeros(points)
    
    ###################################
    
    start_music = (((a[a['class'] == 'Music']['start min'].dropna()*60)+(a[a['class'] == 'Music']['start sec'].dropna()))/t).values
    start_music = np.round(start_music,0)
    end_music = (((a[a['class'] == 'Music']['end min'].dropna()*60)+(a[a['class'] == 'Music']['end sec'].dropna()))/t).values
    end_music = np.round(end_music,0)
    
    ##################################
    
    start_speech = (((a[a['class'] == 'Speech']['start min'].dropna()*60)+(a[a['class'] == 'Speech']['start sec'].dropna()))/t).values
    start_speech = np.round(start_speech,0)
    end_speech = (((a[a['class'] == 'Speech']['end min'].dropna()*60)+(a[a['class'] == 'Speech']['end sec'].dropna()))/t).values
    end_speech = np.round(end_speech,0)
    
    ##################################
    
    start_silence = (((a[a['class'] == 'Silence']['start min'].dropna()*60)+(a[a['class'] == 'Silence']['start sec'].dropna()))/t).values
    start_silence = np.round(start_silence,0)
    end_silence = (((a[a['class'] == 'Silence']['end min'].dropna()*60)+(a[a['class'] == 'Silence']['end sec'].dropna()))/t).values
    end_silence = np.round(end_silence,0)
    
    ######################################
    
    #Generating music data
    try:
        music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                 np.arange(start_music[1],end_music[1]),
                                np.arange(start_music[2],end_music[2]),
                                np.arange(start_music[3],end_music[3]),
                                np.arange(start_music[4],end_music[4])))
    except (IndexError,ValueError) as e:
        try:
            music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                 np.arange(start_music[1],end_music[1]),
                                np.arange(start_music[2],end_music[2]),
                                np.arange(start_music[3],end_music[3])))
        
        except (IndexError,ValueError) as e:
            try:
                music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                 np.arange(start_music[1],end_music[1]),
                                np.arange(start_music[2],end_music[2])))
            except (IndexError,ValueError) as e:
                try:
                    music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                 np.arange(start_music[1],end_music[1])))
                except (IndexError,ValueError) as e:
                    try:
                        music = np.arange(start_music[0],end_music[0]) 
                    except ValueError:
                        pass
    try:
        music = music.astype('int16')
        for i in music:
            tot[i] = 1
    except NameError:
        pass
    
    
    #######################################
        
    #Generating speech data
    try:
        speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                 np.arange(start_speech[1],end_speech[1]),
                                np.arange(start_speech[2],end_speech[2]),
                                np.arange(start_speech[3],end_speech[3]),
                                np.arange(start_speech[4],end_speech[4])))
    except (IndexError,ValueError) as e:
        try:
            speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                 np.arange(start_speech[1],end_speech[1]),
                                np.arange(start_speech[2],end_speech[2]),
                                np.arange(start_speech[3],end_speech[3])))
        
        except (IndexError,ValueError) as e:
            try:
                speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                 np.arange(start_speech[1],end_speech[1]),
                                np.arange(start_speech[2],end_speech[2])))
            except (IndexError,ValueError) as e:
                try:
                    speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                 np.arange(start_speech[1],end_speech[1])))
                except (IndexError,ValueError) as e:
                    try:
                        speech = np.arange(start_speech[0],end_speech[0]) 
                    except ValueError:
                        pass
                    
    try:
        speech = speech.astype('int16')
        for i in speech:
            tot[i] = 2
    except NameError:
        pass
    
    ######################################
    
    #Generating silence data
    try:
        silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                 np.arange(start_silence[1],end_silence[1]),
                                np.arange(start_silence[2],end_silence[2]),
                                np.arange(start_silence[3],end_silence[3]),
                                np.arange(start_silence[4],end_silence[4])))
    
    except (IndexError,ValueError) as e:
        try:
            silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                 np.arange(start_silence[1],end_silence[1]),
                                np.arange(start_silence[2],end_silence[2]),
                                np.arange(start_silence[3],end_silence[3])))
        except (IndexError,ValueError) as e:
            try:
                silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                 np.arange(start_silence[1],end_silence[1]),
                                np.arange(start_silence[2],end_silence[2])))
            except (IndexError,ValueError) as e:
                try:
                    silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                 np.arange(start_silence[1],end_silence[1])))
                except (IndexError,ValueError) as e:
                    try:
                        silence = np.arange(start_silence[0],end_silence[0]) 
                    except ValueError:
                        pass
                    
    try:
        silence = silence.astype('int16')
        for i in silence:
            tot[i] = 3
    except NameError:
        pass
    
    if total_pts[0] == int(total_pts[0]):
        pass
    else:
        tot = tot[:-1]
    return tot






###############################################################################################
def new_samples(t,n_audio):    
    
    y_new = np.array([])
    for i in range(n_audio):
        
        a = pd.read_excel('data.xlsx', str(i))
        total_pts = ((a['Total min'].dropna()*60)+(a['Total sec'].dropna()))/t
        points = ceil(total_pts)
        tot = np.zeros(points)
        
        ###################################
        
        start_music = (((a[a['class'] == 'Music']['start min'].dropna()*60)+(a[a['class'] == 'Music']['start sec'].dropna()))/t).values
        start_music = np.round(start_music,0)
        end_music = (((a[a['class'] == 'Music']['end min'].dropna()*60)+(a[a['class'] == 'Music']['end sec'].dropna()))/t).values
        end_music = np.round(end_music,0)
        
        ##################################
        
        start_speech = (((a[a['class'] == 'Speech']['start min'].dropna()*60)+(a[a['class'] == 'Speech']['start sec'].dropna()))/t).values
        start_speech = np.round(start_speech,0)
        end_speech = (((a[a['class'] == 'Speech']['end min'].dropna()*60)+(a[a['class'] == 'Speech']['end sec'].dropna()))/t).values
        end_speech = np.round(end_speech,0)
        
        ##################################
        
        start_silence = (((a[a['class'] == 'Silence']['start min'].dropna()*60)+(a[a['class'] == 'Silence']['start sec'].dropna()))/t).values
        start_silence = np.round(start_silence,0)
        end_silence = (((a[a['class'] == 'Silence']['end min'].dropna()*60)+(a[a['class'] == 'Silence']['end sec'].dropna()))/t).values
        end_silence = np.round(end_silence,0)
        
        ######################################
        
        #Generating music data
        try:
            music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2]),
                                    np.arange(start_music[3],end_music[3]),
                                    np.arange(start_music[4],end_music[4])))
        except (IndexError,ValueError) as e:
            try:
                music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2]),
                                    np.arange(start_music[3],end_music[3])))
            
            except (IndexError,ValueError) as e:
                try:
                    music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1]),
                                    np.arange(start_music[2],end_music[2])))
                except (IndexError,ValueError) as e:
                    try:
                        music = np.concatenate((np.arange(start_music[0],end_music[0]),
                                     np.arange(start_music[1],end_music[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            music = np.arange(start_music[0],end_music[0]) 
                        except ValueError:
                            pass
        try:
            music = music.astype('int16')
            for i in music:
                tot[i] = 1
        except NameError:
            pass
        
        
        #######################################
            
        #Generating speech data
        try:
            speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2]),
                                    np.arange(start_speech[3],end_speech[3]),
                                    np.arange(start_speech[4],end_speech[4])))
        except (IndexError,ValueError) as e:
            try:
                speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2]),
                                    np.arange(start_speech[3],end_speech[3])))
            
            except (IndexError,ValueError) as e:
                try:
                    speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1]),
                                    np.arange(start_speech[2],end_speech[2])))
                except (IndexError,ValueError) as e:
                    try:
                        speech = np.concatenate((np.arange(start_speech[0],end_speech[0]),
                                     np.arange(start_speech[1],end_speech[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            speech = np.arange(start_speech[0],end_speech[0]) 
                        except ValueError:
                            pass
                        
        try:
            speech = speech.astype('int16')
            for i in speech:
                tot[i] = 2
        except NameError:
            pass
        
        ######################################

        #Generating silence data
        try:
            silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2]),
                                    np.arange(start_silence[3],end_silence[3]),
                                    np.arange(start_silence[4],end_silence[4])))
        
        except (IndexError,ValueError) as e:
            try:
                silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2]),
                                    np.arange(start_silence[3],end_silence[3])))
            except (IndexError,ValueError) as e:
                try:
                    silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1]),
                                    np.arange(start_silence[2],end_silence[2])))
                except (IndexError,ValueError) as e:
                    try:
                        silence = np.concatenate((np.arange(start_silence[0],end_silence[0]),
                                     np.arange(start_silence[1],end_silence[1])))
                    except (IndexError,ValueError) as e:
                        try:
                            silence = np.arange(start_silence[0],end_silence[0]) 
                        except ValueError:
                            pass
                        
        try:
            silence = silence.astype('int16')
            for i in silence:
                tot[i] = 3
        except NameError:
            pass
        
        if total_pts[0] == int(total_pts[0]):
            y_new = np.append(y_new,tot)
        else:
            tot = tot[:-1]
            y_new = np.append(y_new,tot)
    return y_new
####################################################################################################

