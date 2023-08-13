import leaf_audio.frontend as frontend
import tensorflow as tf
import os
import librosa
import matplotlib.pyplot as plt

leaf = frontend.Leaf()


# melfbanks = frontend.MelFilterbanks()
# tfbanks = frontend.TimeDomainFilterbanks()
# sincnet = frontend.SincNet()
# sincnet_plus = frontend.SincNetPlus()


def reafpic(audiofilespath):
    classes = os.listdir(audiofilespath)  # 类列表
    path = 'picture'
    for each in classes:  # 取类
        path2 = os.path.join(path, each)
        if not os.path.exists(path2):
            os.makedirs(path2)
        class_path = audiofilespath + each
        directory_in_str = class_path
        directory = os.fsencode(directory_in_str)  # 目录编码
        for file in os.listdir(directory):  # 这里遍历类中的wav
            filename = os.fsdecode(file)  # 目录解码
            if filename.endswith(".wav"):
                file_path = os.path.join(directory_in_str, filename)
                data, sr = librosa.load(file_path, sr=None)
                audio_sample = data[tf.newaxis, :]
                leaf_representation = leaf(audio_sample)
                print(leaf_representation.shape)
                i = 0
                for n in range(leaf_representation.shape[1] // 40):
                    result = leaf_representation[0][i:i + 40]
                    i += 40
                    # print(result.shape)
                    # 帧长560，帧移400
                    plt.figure(figsize=(1, 1))
                    plt.imshow(result)
                    plt.axis('off')
                    # plt.show()
                    plt.savefig(path2 + '/' + each + str(n) + '.jpg')
                    plt.close()


# dataset = iter(tfds.load('speech_commands', split='train', shuffle_files=True))
# # Audio is in int16, we rescale it to [-1; 1].
# audio_sample = next(dataset)['audio'] / tf.int16.max
# # The frontend expects inputs of shape [B, T] or [B, T, C].
# audio_sample = audio_sample[tf.newaxis, :]
# leaf_representation = leaf(audio_sample)
# melfbanks_representation = melfbanks(audio_sample)
# tfbanks_representation = tfbanks(audio_sample)
# sincnet_representation = sincnet(audio_sample)
# sincnet_plus_representation = sincnet_plus(audio_sample)

if __name__ == '__main__':
    path = 'data2/'
    reafpic(path)
