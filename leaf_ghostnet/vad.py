import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import wave


class VoiceActivityDetector():
    """ Use signal energy to detect voice activity in wav file """

    def __init__(self, wave_input_filename):
        # self._read_wav(wave_input_filename)._convert_to_mono()
        self._audioread(wave_input_filename)
        self.sample_window = 0.02  # 20 ms
        self.sample_overlap = 0.01  # 10ms
        self.speech_window = 0.5  # half a second
        # self.speech_energy_threshold = 0.6  # 60% of energy in voice band
        self.speech_energy_threshold = 0.2  # 60% of energy in voice band
        # self.speech_start_band = 300
        # self.speech_end_band = 3000
        self.speech_start_band = 1000
        self.speech_end_band = 6000

    def _read_wav(self, wave_file):
        self.rate, self.data = wf.read(wave_file)
        self.channels = len(self.data.shape)
        self.filename = wave_file
        return self

    def _audioread(self, file_path):
        # Load audio file at its native sampling rate
        f = wave.open(file_path, 'rb')
        params = f.getparams()
        nchannels, sampwidth, samplerate, nframes = params[:4]  # nframes就是点数
        # print("read audio dimension", nchannels, sampwidth, samplerate, nframes)
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = np.reshape(waveData, [nframes, nchannels]).T
        self.data = waveData[0, :]
        self.data = self.data / abs(self.data).max()  # 对语音进行归一化
        # print("read audio size", data.shape)
        f.close()
        self.rate = samplerate
        return self

    def _convert_to_mono(self):
        if self.channels == 2:
            self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
            self.channels = 1
        return self

    def _calculate_frequencies(self, audio_data):
        data_freq = np.fft.fftfreq(len(audio_data), 1.0 / self.rate)
        data_freq = data_freq[1:]
        return data_freq

    def _calculate_amplitude(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))  # 傅里叶变换
        data_ampl = data_ampl[1:]
        return data_ampl

    def _calculate_energy(self, data):
        data_amplitude = self._calculate_amplitude(data)  # fft
        data_energy = data_amplitude ** 2  # 取平方
        return data_energy

    def _znormalize_energy(self, data_energy):
        energy_mean = np.mean(data_energy)
        energy_std = np.std(data_energy)
        energy_znorm = (data_energy - energy_mean) / energy_std
        return energy_znorm

    def _connect_energy_with_frequencies(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):  # 遍历频率尺度
            if abs(freq) not in energy_freq:  # 0-8000，50
                energy_freq[abs(freq)] = data_energy[i] * 2  # 扩大两倍？
        return energy_freq

    def _calculate_normalized_energy(self, data):
        data_freq = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        # data_energy = self._znormalize_energy(data_energy) #znorm brings worse results
        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)
        return energy_freq

    def _sum_energy_in_band(self, energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():  # 键
            if start_band < f < end_band:  # 若键在范围内
                sum_energy += energy_frequencies[f]  # 添加进去
        return sum_energy

    def _median_filter(self, x, k):
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros((len(x), k), dtype=x.dtype)
        y[:, k2] = x
        for i in range(k2):
            j = k2 - i
            y[j:, i] = x[:-j]
            y[:j, i] = x[0]
            y[:-j, -(i + 1)] = x[j:]
            y[-j:, -(i + 1)] = x[-1]
        return np.median(y, axis=1)

    def _smooth_speech_detection(self, detected_windows):
        median_window = int(self.speech_window / self.sample_window)
        if median_window % 2 == 0: median_window = median_window - 1
        median_energy = self._median_filter(detected_windows[:, 1], median_window)
        return median_energy

    def convert_windows_to_readible_labels(self, detected_windows):
        """ Takes as input array of window numbers and speech flags from speech
        detection and convert speech flags to time intervals of speech.
        Output is array of dictionaries with speech intervals.
        """
        speech_time = []
        mydata = np.array([])
        doub = np.array([])
        is_speech = 0
        for window in detected_windows:
            if window[1] == 1.0 and is_speech == 0:  # 如果是说话，但标识不是
                is_speech = 1
                speech_label = {}
                speech_time_start = window[0] / self.rate  # 计算开始时间
                speech_label['speech_begin'] = speech_time_start
                # print(window[0], speech_time_start)  # 打印起点，以及起点时间
                # speech_time.append(speech_label)
            if window[1] == 0.0 and is_speech == 1:  # 如果不在说话，标识为是
                is_speech = 0
                speech_time_end = window[0] / self.rate  # 计算结束时间
                speech_label['speech_end'] = speech_time_end  # 添加结束时间
                speech_time.append(speech_label)  # 加入speech_time中
                # print(window[0], speech_time_end)
                mydata = np.append(mydata, self.data[int(speech_label['speech_begin']*self.rate): int(speech_label['speech_end']*self.rate)])
                doub = np.append(doub, [int(speech_label['speech_begin']*self.rate), int(speech_label['speech_end']*self.rate)])
                # print(mydata.__len__())
        return speech_time, mydata, doub

    def plot_detected_speech_regions(self):
        """ Performs speech detection and plot original signal and speech regions.
        """
        data = self.data
        detected_windows = self.detect_speech()
        data_speech = np.zeros(len(data))
        it = np.nditer(detected_windows[:, 0], flags=['f_index'])
        while not it.finished:
            data_speech[int(it[0])] = data[int(it[0])] * detected_windows[it.index, 1]
            it.iternext()
        plt.figure()
        plt.plot(data_speech)
        plt.plot(data)
        plt.show()
        return self

    def detect_speech(self):
        """ Detects speech regions based on ratio between speech band energy
        and total energy.
        Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).
        """
        detected_windows = np.array([])
        sample_window = int(self.rate * self.sample_window)  # 11025*0.02=220 共20ms
        sample_overlap = int(self.rate * self.sample_overlap)  # 11025*0.01=110
        data = self.data  # （23872，）min0，max255，unint8
        sample_start = 0
        start_band = self.speech_start_band  # startband 300
        end_band = self.speech_end_band  # endband 3000
        while (sample_start < (len(data) - sample_window)):  # 样本起始点不超过长度23872-220
            sample_end = sample_start + sample_window  # 起始点偏移220，作为终止点
            if sample_end >= len(data): sample_end = len(data) - 1  # 终止点超过长度，则定义为数组最后一个数
            data_window = data[sample_start:sample_end]  # 取一帧
            energy_freq = self._calculate_normalized_energy(data_window)  # 计算频域归一能量
            sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)  # 85105 float64
            sum_full_energy = sum(energy_freq.values())  # 106700 float64
            speech_ratio = sum_voice_energy / sum_full_energy  # 0.79
            # Hipothesis is that when there is a speech sequence we have ratio of energies more than Threshold
            speech_ratio = speech_ratio > self.speech_energy_threshold  # 和能量阈值0.6对比，返回true
            detected_windows = np.append(detected_windows, [sample_start, speech_ratio])  # [起点，是否为说话声]
            sample_start += sample_overlap  # 样本起点偏移110，取下一帧
        detected_windows = detected_windows.reshape(int(len(detected_windows) / 2), 2)  # reshape一下
        detected_windows[:, 1] = self._smooth_speech_detection(detected_windows)  # 平滑
        return detected_windows, self.data
