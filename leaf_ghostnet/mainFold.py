from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
from ghost_model import GhostNet
import wave
import vad



batch_size = 20  # 64,0.9-0.8
trainsize = 2672
testsize = 1054

def choose_windows(name='Hamming', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    return window


def audioread(file_path):
    # Load audio file at its native sampling rate
    f = wave.open(file_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, samplerate, nframes = params[:4]  # nframes就是点数
    # print("read audio dimension", nchannels, sampwidth, samplerate, nframes)
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.frombuffer(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    data = waveData[0, :]
    # print("read audio size", data.shape)
    f.close()
    return data, samplerate


def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧,且去掉过短的信号
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    winfunc窗函数winfunc = signal.hamming(nw)
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
        return None, nf
    else:  # 否则，计算帧的总长度
        nf = int(np.floor((1.0 * signal_length - nw + inc) / inc))
        whole_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
        pro_signal = signal[0: whole_length]  # 截去后的信号记为pro_signal
        indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                               (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
        indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
        frames = pro_signal[indices]  # 得到帧信号
        win = np.tile(winfunc, (nf, 1))  # window窗函数，这里默认取1
        # print("enframe finished")
        return frames * win, nf  # 返回帧信号矩阵


def pretune(co, data):
    # 实现对信号预加重，co为预加重系数，data为加重对象,一维数组.
    size_data = len(data)
    ad_signal = np.zeros(size_data)
    ad_signal[0] = data[0]
    # print(size_data)
    for i in range(1, size_data, 1):
        ad_signal[i] = data[i] - co * data[i - 1]  # 取矩阵中的数值用方括号
    return ad_signal


def dsilence(data, alfa, samplerate):
    # 去除data的静音区，alfa为能量门限值，frame_length分帧长度,, hop_length帧偏移

    data = pretune(0.955, data)
    edata = data / (abs(data).max())  # 对语音进行归一化
    frame_length = int(2000 * samplerate / 1000)  # 50ms帧长
    winfunc = choose_windows('Hanning', frame_length)
    # winfunc = sg.hamming(frame_length)
    frames, nf = enframe(edata, frame_length, frame_length, winfunc)
    if nf != 1:
        frames = frames.T

        # 要以分割得到的帧数作为row
        row = frames.shape[1]  # 帧数
        col = frames.shape[0]  # 帧长

        # print('帧数',frames.shape)
        Energy = np.zeros((1, row))

        # 短时能量函数
        for i in range(0, row):
            Energy[0, i] = np.sum(abs(frames[:, i] * frames[:, i]), 0)  # 不同分帧函数这里要换

        Ave_Energy = Energy.sum() / row
        Delete = np.zeros((1, row))

        # Delete(i)=1 表示第i帧为清音帧

        for i in range(0, row):
            if Energy[0, i] < Ave_Energy * alfa:
                Delete[0, i] = 1

        # 保存去静音的数据
        ds_data = np.zeros((frame_length * int(row - Delete.sum())))

        begin = 0
        for i in range(0, row - 1):
            if Delete[0, i] == 0:
                for j in range(0, frame_length, 1):
                    ds_data[begin * frame_length + j] = edata[i * frame_length + j]
                begin = begin + 1
        ifdata = 1
        return edata, ds_data, ifdata
    else:
        ifdata = 0
        return edata, None, ifdata


def slicewav(audiofilespath):
    # 保存剪切的音频'E:/基金/fastchirplet-master/fastchirplet-master/audio/'
    # 音频文件路径，帧长时间单位，帧的重叠率
    print('载入音频')
    print('---------------------------------')
    frame_time = 4000
    overlap_rate = 0.5
    classes = os.listdir(audiofilespath)  # 类列表

    allFrames = None
    allLabels = []
    allEach = []

    for label, each in enumerate(classes):
        num_each = 0
        class_path = audiofilespath + each
        directory_in_str = class_path
        directory = os.fsencode(directory_in_str)  # 目录编码
        for file in os.listdir(directory):  # 这里遍历类中的wav
            filename = os.fsdecode(file)  # 目录解码
            if filename.endswith(".wav"):
                file_path = os.path.join(directory_in_str, filename)
                data, sr = audioread(file_path)  # 1053696, float32
                assert sr == 16000
                frame_length = int(sr * frame_time / 1000)  # 这里使用的是0.3秒为一个帧长
                hop_length = int((1 - overlap_rate) * frame_length)  # 帧移，重叠率越高帧移越短
                ds_data, dedata, ifdata = dsilence(data, 0.1, sr)  # 去除静音区，0.6是无声阈值，越高留下越小


                ifdata = 1
                if ifdata != 0:
                    # 按照文件长度分帧，防止能量泄漏，加窗
                    winfunc = choose_windows('Hanning', frame_length)

                    frames, nf = enframe(dedata, frame_length, hop_length, winfunc)
                    if nf != 1:
                        if allFrames is None:
                            allFrames = frames  # [12, 15360]
                            allLabels = np.full((frames.shape[0],), label)
                        else:
                            allFrames = np.vstack((allFrames, frames))
                            allLabels = np.append(allLabels, np.full((frames.shape[0],), label))
                        num_each += len(frames)
        print(str(each) + '    num: ' + str(num_each))
        allEach.append(num_each)
    # np.save('doub.npy', double)
    index = [i for i in range(len(allLabels))]
    np.random.seed(123)
    np.random.shuffle(index)
    train_audio = np.asarray(allFrames)
    train_labels = np.asarray(allLabels)
    train_audio = train_audio[index]
    train_labels = train_labels[index]
    print(train_audio.shape)
    print(train_labels.shape)
    return train_audio,train_labels

if __name__ == '__main__':
    # data_dir = 'E:\\temp3\\clean_train\\'
    data_dir = 'E:\\data\\covid\\coughnew\\val\\'
    data_dir2 = 'E:\\data\\covid\\coughnew\\test\\'

    input_train, target_train = slicewav(data_dir)
    input_test, target_test =slicewav(data_dir2)
#input_test,target_test = input_fn('test73.tfrecords')

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)

print(inputs.shape)
print(targets.shape)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    # Define the model architecture
    model = GhostNet(2)

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  # 内置的loss函数
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=0.001),  # , decay=0.01
        # optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule),#, decay=0.01
        metrics=['accuracy']
    )  # 模型编译
    model.summary_model()
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(inputs[train], targets[train], epochs=30, steps_per_epoch=trainsize // batch_size,
                        validation_steps=testsize // batch_size, callbacks=None, shuffle=True)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])



    # Increase fold number
    fold_no = fold_no + 1