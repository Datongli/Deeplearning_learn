# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : utils.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/20 下午3:19
import wave
import librosa
import numpy as np
import matplotlib.pyplot as plt


def tensor_to_img(spectrogram, x_range=None, y_range=None):
    plt.figure()  # arbitrary, looks good on my screen.
    # plt.imshow(spectrogram[0].T)
    plt.imshow(spectrogram.T)
    if x_range is not None:
        plt.xlim(0, x_range)
    if y_range is not None:
        plt.ylim(0, y_range)
    plt.show()


# 绘制频谱图
def plot_spectrogram(spec, note):
    """
    audio feature figure
    (feature_dim, time_step)
    """
    fig = plt.figure(figsize=(20, 5))  # Width, height in inches.
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()


def save_wav(file_name, audio_data, channels=1, sample_width=2, rate=16000):
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(b''.join(audio_data))
    wf.close()


def read_wave_from_file(file_path):
    # Load audio file at its native sampling rate
    f = wave.open(file_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, samplerate, nframes = params[:4]  # nframes就是点数
    # print("read audio dimension", nchannels, sampwidth, samplerate, nframes)
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    data = waveData[0, :]
    # print("read audio size", data.shape)
    f.close()
    return data, samplerate


def concat_frame(features, left_context_width, right_context_width):
    time_steps, features_dim = features.shape
    concated_features = np.zeros(
        shape=[time_steps, features_dim *
               (1 + left_context_width + right_context_width)],
        dtype=np.float32)
    # middle part is just the uttarnce （128， 3+1+0*474）其中1*474放入原feature，
    # 前3中，1取[1:128]放feature[0:127], 2取[2:128]放[0:126], 3取[3:128]放[0:125]
    # 后3中，1取[0:127]放feature[1:128], 2取[0:126]放[2:128], 3取[0:125]放[3:128]
    concated_features[:, left_context_width * features_dim:
                         (left_context_width + 1) * features_dim] = features

    for i in range(left_context_width):
        # add left context
        concated_features[i + 1:time_steps,
        (left_context_width - i - 1) * features_dim:
        (left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

    for i in range(right_context_width):
        # add right context
        concated_features[0:time_steps - i - 1,
        (right_context_width + i + 1) * features_dim:
        (right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

    return concated_features


def subsampling(features, subsample=3):
    interval = subsample
    temp_mat = [features[i]
                for i in range(0, features.shape[0], interval)]
    subsampled_features = np.row_stack(temp_mat)
    return subsampled_features


def get_feature(wave_data, framerate=16000, feature_dim=128):
    """
    :param wave_data: 一维numpy,dtype=int16
    :param framerate:
    :param feature_dim:
    :return: specgram [序列长度,特征维度]
    """
    wave_data = wave_data.astype("float32")
    specgram = librosa.feature.melspectrogram(wave_data, sr=framerate, n_fft=512, hop_length=160, n_mels=feature_dim)
    specgram = np.where(specgram == 0, np.finfo(float).eps, specgram)
    specgram = np.log10(specgram)
    return specgram


def get_final_feature(samples, sample_rate=16000, feature_dim=128, left=3, right=0, subsample=3):
    feature = get_feature(samples, sample_rate, feature_dim)
    feature = concat_frame(feature, left, right)
    feature = subsampling(feature, subsample)  # 将128滤波器组按照3来跳着取
    return feature  # 43，1896  频域上的数据增强


def log_mel(file, sr=16000, dim=80, win_len=25, stride=10):
    samples, sr = librosa.load(file, sr=sr)
    samples = samples * 32768
    win_len = int(sr / 1000 * win_len)
    hop_len = int(sr / 1000 * stride)
    feature = librosa.feature.melspectrogram(samples, sr=sr, win_length=win_len, hop_length=hop_len, n_mels=dim)
    feature = np.where(feature == 0, np.finfo(float).eps, feature)  # 查找替换，eps是取非负的最小值
    feature = np.log10(feature)
    return feature


if __name__ == '__main__':
    path = 'audio/speech.wav'
    data, rate = read_wave_from_file(path)
    get_final_feature(data)
