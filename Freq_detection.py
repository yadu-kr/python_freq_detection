# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:58:14 2021

@author: yadukr97
"""
import pyaudio
import time
import numpy as np
import wave

chunk = 10000

WIDTH = 2
CHANNELS = 1
RATE = 44100

# use a Blackman window
window = np.blackman(chunk)

p = pyaudio.PyAudio()


stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                frames_per_buffer=chunk,
                input=True)

stream.start_stream()

while stream.is_active():
    time.sleep(0.5)
    
    try:
        
        data = stream.read(10000,exception_on_overflow=False)
        # unpack the data and times by the hamming window
        indata = np.array(wave.struct.unpack("%dh"%(len(data)/WIDTH),data))*window
        # Take the fft and square each value
        fftData=abs(np.fft.rfft(indata))**2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(fftData)-1:
            y0,y1,y2 = np.log(fftData[which-1:which+2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it 
            thefreq = (which+x1)*RATE/chunk
            print("The freq is %f Hz." % (thefreq))
        else:
            thefreq = which*RATE/chunk
            print ("The freq is %f Hz." % (thefreq))
   
            
    except KeyboardInterrupt:
            break

stream.stop_stream()
stream.close()

p.terminate()
