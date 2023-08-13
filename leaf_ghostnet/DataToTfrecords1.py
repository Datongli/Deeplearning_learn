import tensorflow as tf
# import librosa
import numpy as np
import os
import wave
import vad

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    print('正在生成tfrecord')
    print('-------------------------------------------------')
    frame_time = 4000
    overlap_rate = 0.5
    classes = os.listdir(audiofilespath)  # 类列表

    split = 0
    split2 = 0
    allFrames = None
    allFrames_shift = None
    allLabels = []
    allLabels_shift = []
    allEach = []
    double = {}

    with tf.io.TFRecordWriter("./trainnew.tfrecords") as writer:  # tfrecords功能
        with tf.io.TFRecordWriter("./testnew.tfrecords") as writer2:
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

                        # # 频域vad
                        # tdata = pretune(0.955, data)
                        # dedata = tdata / abs(tdata).max()  # 对语音进行归一化

                        # v = vad.VoiceActivityDetector(file_path)
                        # raw_detection, initdata = v.detect_speech()
                        # speech_labels, dedata, doub = v.convert_windows_to_readible_labels(raw_detection)

                        # # 保存npy
                        # double[file_path] = doub

                        ifdata = 1
                        # ds_data=float32,1053696  dedata=float64,87040
                        if ifdata != 0:
                            # 按照文件长度分帧，防止能量泄漏，加窗
                            winfunc = choose_windows('Hanning', frame_length)
                            # if each == 'others':
                            #     frames, nf = enframe(ds_data, frame_length, hop_length, winfunc)
                            # else:
                            #     frames, nf = enframe(dedata, frame_length, hop_length, winfunc)
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
            # print(train_audio)
            for i in range(len(train_labels)):
                #if i % 5 != 0:
                save_tfrecords(train_audio[i], train_labels[i], writer=writer)  # 按帧写入tfrecords
                split += 1
                # if i % 2.5 >= 1:
                #     save_tfrecords(train_audio[i], train_labels[i], writer=writer)  # 按帧写入tfrecords
                #     split += 1
                # else:
                #     save_tfrecords(train_audio[i], train_labels[i], writer=writer2)  # 按帧写入tfrecords
                #     split2 += 1

            # # shift循环
            # print("循环增强---------------")
            # for label, each in enumerate(classes):
            #     num_each = 0
            #     directory_in_str = audiofilespathshift + each
            #     directory = os.fsencode(directory_in_str)  # 目录编码
            #     for file in os.listdir(directory):  # 这里遍历类中的wav
            #         filename = os.fsdecode(file)  # 目录解码
            #         if filename.endswith(".wav"):
            #             file_path = os.path.join(directory_in_str, filename)
            #             data, sr = audioread(file_path)  # 1053696, float32
            #             frame_length = int(sr * frame_time / 1000)  # 这里使用的是0.3秒为一个帧长
            #             hop_length = int((1 - overlap_rate) * frame_length)  # 帧移，重叠率越高帧移越短
            #             # ds_data, dedata, ifdata = dsilence(data, 0.1, sr)  # 去除静音区，0.6是无声阈值，越高留下越小
            #
            #             # 频域vad
            #             tdata = pretune(0.955, data)
            #             ds_data = tdata / abs(tdata).max()  # 对语音进行归一化
            #             v = vad.VoiceActivityDetector(file_path)
            #             raw_detection, initdata = v.detect_speech()
            #             speech_labels, dedata, doub = v.convert_windows_to_readible_labels(raw_detection)
            #             ifdata = 1
            #
            #             # ds_data=float32,1053696  dedata=float64,87040
            #             if ifdata != 0:
            #                 # 按照文件长度分帧，防止能量泄漏，加窗
            #                 winfunc = choose_windows('Hanning', frame_length)
            #                 if each == 'others':
            #                     frames, nf = enframe(ds_data, frame_length, hop_length, winfunc)
            #                 else:
            #                     frames, nf = enframe(dedata, frame_length, hop_length, winfunc)
            #                 if nf != 1:
            #                     if allFrames_shift is None:
            #                         allFrames_shift = frames  # [12, 15360]
            #                         allLabels_shift = np.full((frames.shape[0],), label)
            #                     else:
            #                         allFrames_shift = np.vstack((allFrames_shift, frames))
            #                         # allLabels = np.vstack((allLabels, np.full((frames.shape[0],), label)))
            #                         allLabels_shift = np.append(allLabels_shift, np.full((frames.shape[0],), label))
            #                     num_each += len(frames)
            #     allEach[label] += num_each
            #     print(str(each) + '    num: ' + str(allEach[label]))
            # index = [i for i in range(len(allLabels_shift))]
            # np.random.shuffle(index)
            # train_audio = np.asarray(allFrames_shift)
            # train_labels = np.asarray(allLabels_shift)
            # train_audio = train_audio[index]
            # train_labels = train_labels[index]
            # # print(train_audio)
            #
            # # shift滚动
            # data_type = train_audio[0].dtype
            # max_shifts = train_audio.shape[1] * 0.05
            # nb_shifts = np.random.randint(-max_shifts, max_shifts)
            # train_audio = np.roll(train_audio, nb_shifts, axis=1)
            # train_audio = train_audio.astype(data_type)
            #
            # for i in range(len(train_labels)):
            #     if i % 5 != 0:
            #         save_tfrecords(train_audio[i], train_labels[i], writer=writer)  # 按帧写入tfrecords
            #         split += 1
            #     else:
            #         save_tfrecords(train_audio[i], train_labels[i], writer=writer2)  # 按帧写入tfrecords
            #         split2 += 1

            lossnum = allEach

            print(split, split2)
            lossnum2 = []
            for i, each in enumerate(lossnum):
                lossnum2.append([])
                lossnum2[i].append(round((1. - (each / (split + split2))), 2))
            print("loss alpha：" + str(lossnum2))


def save_tfrecords(data, label, writer):
    features = tf.train.Features(
        feature={
            "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.astype(np.float32).tobytes()])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
    )
    example = tf.train.Example(features=features)
    serialized = example.SerializeToString()
    writer.write(serialized)


if __name__ == '__main__':
    # data_dir = 'E:\\temp3\\clean_train\\'
    data_dir = 'E:\\code\\leaf_ghostnet\\data\\coughnew\\train\\'

    # data_dir2 = 'data_two\\'
    # data_dir = 'data11\\'
    # data_dir2 = 'datahard2\\'
    slicewav(data_dir)
