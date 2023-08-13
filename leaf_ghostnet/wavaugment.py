import librosa
import random
import numpy as np
import os
from utils2 import read_wave_from_file, save_wav, tensor_to_img, get_feature, plot_spectrogram


# 音调增强
def pitch_librosa(samples, sr=16000, ratio=5):
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    samples = samples.astype('float')
    ratio = random.uniform(-ratio, ratio)
    samples = librosa.effects.pitch_shift(samples, sr, n_steps=ratio)
    samples = samples.astype(data_type)
    return samples


# 速度增强
def speed_numpy(samples, speed=None, min_speed=0.9, max_speed=1.1):
    """
    numpy线形插值速度增益
    :param speed: 速度
    :param samples: 音频数据，一维
    :param max_speed: 不能低于0.9，太低效果不好
    :param min_speed: 不能高于1.1，太高效果不好
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    if speed is None:
        speed = random.uniform(min_speed, max_speed)
    old_length = samples.shape[0]
    new_length = int(old_length / speed)
    old_indices = np.arange(old_length)  # (0,1,2,...old_length-1)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)  # 在指定的间隔内返回均匀间隔的数字
    samples = np.interp(new_indices, old_indices, samples)  # 一维线性插值
    samples = samples.astype(data_type)
    return samples


# 音量增强
def volume_augment(samples, min_gain_dBFS=-10, max_gain_dBFS=10):
    """
    音量增益范围约为【0.316，3.16】，不均匀，指数分布，降低幂函数的底10.可以缩小范围
    :param samples: 音频数据，一维
    :param min_gain_dBFS:
    :param max_gain_dBFS:
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    rate = 16000
    myframe = len(samples) // rate
    data_type = samples[0].dtype
    for i in range(myframe):
        gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
        gain = 10. ** (gain / 20.)
        samples[i*rate:(i+1)*rate] = samples[i*rate:(i+1)*rate] * gain
    # improvement:保证输出的音频还是原类型，不然耳朵会聋
    samples = samples.astype(data_type)
    return samples


def data_aug(name, augnum):
    path = 'datahard2\\'
    # path2 = name + '\\'
    path2 = "data_aughard\\"
    oneaug = 2
    classes = os.listdir(path)  # 类列表
    for each in classes:
        if each == "heiling":
            class_path = path + each
            class_path2 = path2 + each
            if not os.path.exists(class_path2):
                os.makedirs(class_path2)
            for file in os.listdir(class_path):  # 遍历wav
                if file.endswith(".wav"):
                    for i in range(augnum):
                        file_path = os.path.join(class_path, file)
                        # print(file_path)
                        out_file = os.path.join(class_path2, file)
                        out_file = str(out_file.split('.wav')[0]) + name + str(i+oneaug) + '.wav'
                        print(out_file)
                        # out_file = str(out_file.split('.wav')[0])+path2.split('data')[-1].split('\\')[0]+'.wav'
                        audio_data, frame_rate = read_wave_from_file(file_path)
                        if name == 'pitch':
                            audio_data = pitch_librosa(audio_data)  # 音调增强
                            save_wav(out_file, audio_data)
                        elif name == 'speed':
                            audio_data = speed_numpy(audio_data, 0.9)  # 语速增强
                            save_wav(out_file, audio_data)
                        elif name == 'volume':
                            audio_data = volume_augment(audio_data)  # 音响度增强
                            save_wav(out_file, audio_data)
                        else:
                            print("命名错误")


if __name__ == '__main__':
    data_aug("pitch", 2)
    data_aug("speed", 2)
    print("增强完成")

