import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt 
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

#Smoothing
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."

    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


#Video_Duration
def getLength(filename):
  result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]

#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=3):
    s=src.copy()
    pyramid=[s]
    for i in range(level):
        s=cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid


#load video from file
def load_video(video_filename):
    cap=cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_tensor=np.zeros((frame_count,height,width,3),dtype='float')
    x=0
    while cap.isOpened():
        ret,frame=cap.read()
        if ret is True:
            video_tensor[x]=frame
            x+=1
        else:
            break
    return video_tensor,fps

# apply temporal ideal bandpass filter to gaussian video
def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor,levels=3):
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
        vid_data[i]=gaussian_frame
    return vid_data

#amplify the video
def amplify_video(gaussian_vid,amplification=50):
    return gaussian_vid*amplification

#reconstract video from original video and gaussian video
def reconstract_video(amp_video,origin_video,levels=3):
    final_video=np.zeros(origin_video.shape)
    pix_val=[]
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img=img+origin_video[i]
        pix_val.append(tuple(img.mean(axis=0).mean(axis=0)))
        final_video[i]=img
    return final_video,pix_val

#save video to files
def save_video(video_tensor):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("out/4out.avi", fourcc, 30, (width, height), 1)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()

#magnify color
def get_pulse(video_name,low,high,levels=3,amplification=20):
	t,f=load_video(video_name)
	str1 = video_name.split('/')
	str2 = str1[1].split('.')
	str3 = str2[0].split('_')
	clip = VideoFileClip(video_name) # <= !!!
	vtime = clip.duration

	gau_video=gaussian_video(t,levels=levels)
	filtered_tensor=temporal_ideal_filter(gau_video,low,high,f)
	amplified_video=amplify_video(filtered_tensor,amplification=amplification)
	final,lst=reconstract_video(amplified_video,t,levels=3)
	rpix_val=np.zeros(len(lst))
	gpix_val=np.zeros(len(lst))
	bpix_val=np.zeros(len(lst))
	sum_val=np.zeros(len(lst))
	

	for j in range(len(lst)):
		rpix_val[j]= lst[j][0]
		gpix_val[j]= lst[j][1]
		bpix_val[j]= lst[j][2]
		sum_val[j]=lst[j][0]+lst[j][1]+lst[j][2]


	rpix_val = smooth(rpix_val,10,'hanning')
	gpix_val = smooth(gpix_val,10,'hanning')
	bpix_val = smooth(bpix_val,10,'hanning')

	step = vtime/(bpix_val.shape[0])
	time = np.arange(0, vtime, step)
	bmaximas_ind = argrelextrema(bpix_val, np.greater)
	maximas_count = bmaximas_ind[0].shape
	print(maximas_count[0])
	mpulse = (maximas_count[0]*60.0)/(2.0*vtime)
	mpulse = (maximas_count[0]*60.0)/(2.0*vtime)
	mpulse = (maximas_count[0]*60.0)/(2.0*vtime)

	plt.plot(time, rpix_val, color='red')
	plt.plot(time, gpix_val, color='green')
	plt.plot(time, bpix_val, color='blue')  
	#plt.plot(time, sum_val, color='black') 

  
    # setting x and y axis range 
    #plt.ylim(1,8) 
    #plt.xlim(1,8) 
  
    # naming the x axis 
	plt.xlabel('time') 
    # naming the y axis 
	plt.ylabel('Blue component ') 

    # giving a title to my graph 
	plt.title('RGB components variation!  '+'#'+str(str3[0])+', opulse:'+str(str3[1])+', maximas:'+str(maximas_count[0])+', mpulse:'+str(mpulse), fontsize=10)
	plt.savefig('#'+str(str3[0])+'After_smoothing.pdf')
  
    # function to show the plot 
	plt.show() 
	save_video(final)



if __name__=="__main__":
	path = "data/"
	dirs = os.listdir(path)
	for file in sorted(dirs):
		get_pulse(path+file ,0.4,3)