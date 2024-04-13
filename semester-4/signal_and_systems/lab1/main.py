import numpy as np
import matplotlib.pyplot as plt    
import sounddevice as sd
import soundfile as sf

def get_discrete_signal():
    
    n = int(input("Enter the number of samples: "))
    while n < 1:
        n = int(input("Enter a valid number of samples: "))
    
    x = {}

    for i in range(n):
        try:
            index = int(input(f"Enter the index of sample {i+1}: "))
            value = float(input(f"Enter the value of sample {i+1}: "))
            x[index] = value
        except ValueError:
            print("Invalid input. Please try again.")
            i -= 1
    
    return x

def myConv(x, h):
    y = {}
    for n in range(min(x.keys()) + min(h.keys()), max(x.keys()) + max(h.keys()) + 1):
        y[n] = 0
        for k in x.keys():
            if n - k in h.keys():
                y[n] += x[k] * h[n - k]
    return y

def compareConv(x, h):
    y1 = np.convolve(list(x.values()), list(h.values()))
    y2 = myConv(x, h)

    print("numpy.convolve:", y1)
    print("myConv:", y2)

    figure, axis = plt.subplots(2, 1)
    axis[0].stem(list(y1), label="numpy.convolve")
    axis[1].stem(list(y2.values()), label="myConv")

    axis[0].set_title("numpy.convolve")
    axis[1].set_title("myConv")

    plt.show()


    

def convolution():
    
    print("Enter the discrete signal x[n]:")
    x = get_discrete_signal()
    print("Enter the discrete signal h[n]:")
    h = get_discrete_signal()

    y = myConv(x,h)

    print("x[n]:", x)
    print("h[n]:", h)
    

    compareConv(x, h)

def record_sound(duration = 5):
    fs = 44100
    filename = 'output.wav'

    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()
    print("Recording done.")
    return myrecording


def create_h(M, A):
    h = np.zeros(3000*M + 1)
    h[0] = 1
    for k in range(1, M+1):
        h[3000*k] = A**k * k
    return h

def play_sound(sound):
    print("Playing sound...")
    sd.play(sound)
    sd.wait()
    print("Playing done.")



def sound_conv():

    h1 = create_h(3, 2)
    h2 = create_h(4,2)
    h3 = create_h(5,2)

    record5 = record_sound(5)
    record10 = record_sound(10)



    y1 = np.convolve(record5[:,0], h1)
    y2 = np.convolve(record5[:,0], h2)
    y3 = np.convolve(record5[:,0], h3)

    y4 = np.convolve(record10[:,0], h1)
    y5 = np.convolve(record10[:,0], h2)
    y6 = np.convolve(record10[:,0], h3)


    play_sound(record5)
    play_sound(record10)
    play_sound(y1)
    play_sound(y2)
    play_sound(y3)
    play_sound(y4)
    play_sound(y5)
    play_sound(y6)
    






sound_conv()