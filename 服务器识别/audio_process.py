import numpy as np
import chirplet as ch
import matplotlib.pyplot as plt
import librosa.util
import librosa
import os
import scipy.signal as sg
import soundfile as sf
from vad import filter
from enframe import pretune
from enframe import enframe
import csv
import warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    pass


def split_audio_id(file_path, file_id, frame_time, overlap_rate):
    data, sr = librosa.load(file_path, sr=None)
    frame_length = int(sr * frame_time / 1000)
    hop_length = int((1 - overlap_rate) * frame_length)

    tdata = pretune(0.955, data)
    '''plt.plot(tdata)
    plt.ylabel('tdata')
    plt.show()'''

    winfunc = sg.hamming(frame_length)
    frames = enframe(tdata, frame_length, hop_length, winfunc)
    # frames = librosa.util.frame(tdata, frame_length, hop_length).T
    col = frames.shape[1]  # 1591#不同分帧函数这里要换
    row = frames.shape[0]  # 50
    # print('帧数', row, col)

    new_audio_path = os.path.join(file_path[:file_path.rfind("\\")], file_id)
    filename = file_path.split('\\')[-1]

    create_dir_not_exist(new_audio_path)

    for i in range(0, row, 1):
        file_path_new = os.path.join(new_audio_path,
                                     filename[:-4] + str('-') + str(i) + str('.wav'))  # [:-3]指从后往前减3个
        sf.write(file_path_new, frames[i, :], sr)


def vad(file_path, resample_sr, frame_time, overlap_rate):
    file_dir = file_path[: file_path.rfind("\\")]
    file_name = file_path[file_path.rfind("\\")+1: file_path.rfind('.')]
    resample_dir = os.path.join(file_dir, "resample")
    vad_dir = os.path.join(file_dir, "vad")

    create_dir_not_exist(resample_dir)
    create_dir_not_exist(vad_dir)

    file_list = os.listdir(file_dir)
    if file_path.endswith(".wav"):
        print(file_path)
        resample_path = os.path.join(resample_dir, file_name + "-{}K".format(resample_sr / 1000) + ".wav")

        file, sr = librosa.load(file_path, sr=None)
        if not abs(max(file.min(), file.max())) < 0.005:
            file_resample = librosa.resample(file, sr, resample_sr)
            sf.write(resample_path, file_resample, resample_sr)

            filter(resample_path, vad_dir, expand=False)
            os.remove(resample_path)
        else:
            print("{}为空音频".format(file_path))
            with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(file_path)  # 写入数据

    file_vad = os.path.join(vad_dir, file_name + '-{}K-VAD.wav'.format(resample_sr / 1000))
    split_audio_id(file_vad, file_name, frame_time, overlap_rate)
    os.remove(file_vad)


def compute_spectrogram(signal, sample_rate):
    """
        compute spectrogram from signal
        :param signal:
        :return: matrice representing the spectrum, frequencies corresponding and times
    """
    [spectrum, freqs, times] = plt.mlab.specgram(signal, NFFT=1024, Fs=sample_rate,
                                                 noverlap=512, window=np.hamming(1024))
    spectrum = 10. * np.log10(spectrum)

    return [spectrum, freqs, times]


def compute_TFDF(signal):
    freq = np.size(signal, 0)  # 计算 X 的行数
    time = np.size(signal, 1)  # 计算 X 的列数

    TFDF_array = [[]]
    for m in range(freq - 1):
        TFDF_array.append([])
        for t in range(time - 1):
            dE_FT = signal[m, t + 1] + signal[m + 1, t] - signal[m, t] - signal[m + 1, t + 1]
            TFDF_array[m].append(dE_FT)
    TFDF_array.remove([])
    TFDF = np.array(TFDF_array)

    return TFDF


def plotsft(audiopath):
    # print("--- filename---" ,audiopath)
    # print('绘制sft' )
    data, sr = librosa.load(audiopath, sr=None)
    # edata=data/abs(data).max()#对语音进行归一化
    # print("数据长度采样率",len(data),sr)

    figure, axarr = plt.subplots(1, sharex=False)
    figure.set_size_inches(4, 4)
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    [spectrum, freqs, times] = compute_spectrogram(data, sr)
    index_frequency = np.argmax(freqs)
    mxf = freqs[index_frequency]
    # print("最大频率",mxf)

    axarr.matshow(spectrum[0:index_frequency, :], cmap='jet',
                  origin='lower',
                  extent=(times[0], times[-1], freqs[0], mxf),
                  aspect='auto')
    plt.axis('off')
    # axarr.axes.xaxis.set_ticks_position('bottom')
    # axarr.set_ylabel("Frequency in Hz")
    # axarr.xaxis.grid(which='major', color='Black',
    # linestyle='-', linewidth=0.25)
    # axarr.yaxis.grid(which='major', color='Black',
    # linestyle='-', linewidth=0.25)

    # axarr.set_title('spectrogram')

    # figure.tight_layout()

    figure.savefig(audiopath[:-4] + str('.jpg'), dpi=100)
    # plt.show()

    plt.close('all')


def plotmelspec(audiopath):
    print(audiopath)
    # print("--- filename---" ,audiopath)
    # print('绘制spec' )
    data, sr = librosa.load(audiopath, sr=None)
    # edata=data/abs(data).max()#对语音进行归一化
    # print("数据长度采样率",len(data),sr)

    # cmap = cmx.gray   #jet,parula,gray,rainbow
    figure, axarr = plt.subplots(1, sharex=False)
    # glength=len(data)/sr*80
    figure.set_size_inches(4, 4)  # 输出图大小和时间长度成正比50ms一幅图
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    [spectrum, freqs, times] = compute_spectrogram(data, sr)
    index_frequency = np.argmax(freqs)

    max_frequency = freqs[index_frequency]

    melspec = librosa.feature.mfcc(data, sr, n_mfcc=32, n_fft=1024, hop_length=512)  # 计算mel倒谱

    # print('mel',np.shape(melspec))

    axarr.matshow(melspec[1:index_frequency, :], cmap='jet',
                  origin='lower',
                  extent=(times[0], times[-1], freqs[0], max_frequency),
                  aspect='auto')

    plt.axis('off')
    # axarr.axes.xaxis.set_ticks_position('bottom')
    # axarr.set_ylabel("Frequency in Hz")
    # axarr.xaxis.grid(which='major', color='Black',
    # linestyle='-', linewidth=0.25)
    # axarr.yaxis.grid(which='major', color='Black',
    # linestyle='-', linewidth=0.25)

    # axarr.set_title('melspectrogram')

    # figure.tight_layout()

    mel_folder = audiopath[: audiopath.rfind("\\")] + str(r'\MFCC')

    create_dir_not_exist(mel_folder)
    figure.savefig(mel_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'),
                   dpi=56)
    # plt.show()

    plt.close('all')


def plotlogmelspec(audiopath):
    print(audiopath)

    data, sr = librosa.load(audiopath, sr=None)

    figure, axarr = plt.subplots(1, sharex=False)

    figure.set_size_inches(4, 4)  # 输出图大小和时间长度成正比50ms一幅图
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    [spectrum, freqs, times] = compute_spectrogram(data, sr)
    index_frequency = np.argmax(freqs)

    max_frequency = freqs[index_frequency]

    melspec = librosa.feature.melspectrogram(data, sr, n_fft=1024, hop_length=512, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)
    # print('mel',np.shape(melspec))

    axarr.matshow(logmelspec[1:index_frequency, :], cmap='jet',
                  origin='lower',
                  extent=(times[0], times[-1], freqs[0], max_frequency),
                  aspect='auto')

    plt.axis('off')

    mel_folder = audiopath[: audiopath.rfind("\\")] + str(r'\logMel')

    create_dir_not_exist(mel_folder)
    figure.savefig(mel_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'),
                   dpi=56)
    # plt.show()

    plt.close('all')


def plotTFDF(audiopath):
    print(audiopath)
    # print("--- filename---" ,audiopath)
    # print('绘制spec' )
    data, sr = librosa.load(audiopath, sr=None)
    # edata=data/abs(data).max()#对语音进行归一化
    # print("数据长度采样率",len(data),sr)

    # cmap = cmx.gray   #jet,parula,gray,rainbow
    figure, axarr = plt.subplots(1, sharex=False)
    # glength=len(data)/sr*80
    figure.set_size_inches(4, 4)  # 输出图大小和时间长度成正比50ms一幅图
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    [spectrum, freqs, times] = plt.mlab.specgram(data, NFFT=1024, Fs=sr,
                                                 noverlap=512, window=np.hamming(1024))
    melspec = librosa.feature.melspectrogram(data, sr, n_fft=1024, hop_length=512, n_mels=128)
    logmelspec = librosa.power_to_db(melspec)

    TFDF = compute_TFDF(logmelspec)

    index_frequency = np.argmax(freqs)
    max_frequency = freqs[index_frequency]

    axarr.matshow(TFDF[0:index_frequency, :], cmap='jet',
                  origin='lower',
                  extent=(times[0], times[-1], freqs[0], max_frequency),
                  aspect='auto')

    plt.axis('off')

    mel_folder = audiopath[: audiopath.rfind("\\")] + str(r'\TFDF')

    create_dir_not_exist(mel_folder)
    figure.savefig(mel_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'),
                   dpi=56)
    # plt.show()

    plt.close('all')

def plotchirplet(chirps, audiopath):
    # print("--- filename---" ,audiopath)
    data, sr = librosa.load(audiopath, sr=None)
    # edata=data/abs(data).max()#对语音进行归一化
    # print("数据长度采样率",len(data),sr)
    # print('绘制chirp' )
    # cmap = cmx.gray   #jet,parula,gray,rainbow
    figure, axarr = plt.subplots(1, sharex=False)
    # glength=len(data)/sr*80
    figure.set_size_inches(4, 4)
    # 改变图像边沿大小，参数分别为左下右上，子图间距
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    tabfinal = list(reversed(chirps))

    [spectrum, freqs, times] = compute_spectrogram(data, sr)
    index_frequency = np.argmax(freqs)
    mxf = freqs[index_frequency]
    # print("最大频率",mxf)

    axarr.matshow(tabfinal, cmap='jet',
                  origin='lower',
                  extent=(0, times[-1], freqs[0], mxf),
                  aspect='auto')

    plt.axis('off')
    # axarr.axes.xaxis.set_ticks_position('bottom')
    # axarr.set_ylabel("Frequency in Hz")
    # axarr.xaxis.grid(which='major', color='Black',
    #  linestyle='-', linewidth=0.25)
    # axarr.yaxis.grid(which='major', color='Black',
    #  linestyle='-', linewidth=0.25)

    # axarr.set_title('chirplet')

    # figure.tight_layout()

    ch_folder = audiopath[: audiopath.rfind("\\")] + str(r'\chirplet')

    create_dir_not_exist(ch_folder)
    figure.savefig(ch_folder + audiopath[audiopath.rfind("\\"): audiopath.rfind(".")] + str('.jpg'), dpi=56)
    # plt.show()

    plt.close('all')


def ge_graph(pathfile):
    chirplet = []

    for root, dirs, files in os.walk(pathfile):

        for file in files:

            if file.endswith(".wav"):
                chirplet.append(os.path.join(root, file))

    for file in chirplet:
        # start_time = time.time()
        data, sr = librosa.load(file, sr=None)
        # data=data/abs(data).max()#对语音进行归一化
        ch1 = ch.FCT(sample_rate=sr)
        chirps = ch1.compute(data)
        # print(chirps)
        # plotchirplet(chirps, file)
        # plotmelspec(file)
        plotlogmelspec(file)
        # plotTFDF(file)
        # plotsft(file)
        # joblib.dump(chirps, file[:-3] + 'jl')


file_path = r"C:\Users\ldt20\Desktop\root\aADACozj_cough-16.0K-VAD.wav"
vad(file_path, 16000, 1000, 0.5)
# ge_graph(os.path.join(folder_path, "vad", 'new'))
print("finish!!")
