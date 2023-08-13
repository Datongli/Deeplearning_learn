# coding=gbk
import tensorflow
import wave
import os


def myread(file_path):
    print(file_path)
    f = wave.open(file_path, 'rb')
    params = f.getparams()
    # nchannels, sampwidth, samplerate, nframes = params[:4]  # nframes就是点数


if __name__ == '__main__':
    data_dir = 'data_aug2\\'
    for each in os.listdir(data_dir):

        data_dir2 = os.path.join(data_dir, each)
        for wav in os.listdir(data_dir2):
            #if wav != "hongjiaosun001.wav":
            print(wav)
            data_dir3 = os.path.join(data_dir2, wav)
            myread(data_dir3)
