import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import time
import utils
import librosa

# Load TFLite model and allocate tensors.
tflite_model = tf.lite.Interpreter(model_path="chirpModel.tflite")
tflite_model.allocate_tensors()

# Get input and output tensors.
input_details = tflite_model.get_input_details()
output_details = tflite_model.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # 输入随机数
# print(input_data)  # (1, 224, 224, 3)

path = "test_wav/"
label = 0
contents = os.listdir(path)
result = []
t1 = time.time()
for each in contents:  # wav列表
    # print(contents)
    path2 = os.path.join(path, each)
    wav, sr = librosa.load(path2, sr=None)  # 读取音频
    intervals = librosa.effects.split(wav, top_db=20)  # 去静音区
    wav_output = []
    # [可能需要修改参数] 音频长度 16000 * 秒数
    wav_len = 15360
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    if len(wav_output) < wav_len:
        wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
    wav_frames = librosa.util.frame(x=np.array(wav_output), frame_length=wav_len, hop_length=int(wav_len*0.5), axis=0)
    for i, wav_frame in enumerate(wav_frames):
        # print(image)
        wav_frame = wav_frame[np.newaxis, :]
        tflite_model.set_tensor(input_details[0]['index'], wav_frame)
        tflite_model.invoke()
        # output_data = tflite_model.get_tensor(output_details[0]['index'])[0].tolist()
        output_data = tflite_model.get_tensor(output_details[0]['index'])[0][0].tolist()
        # print("out_class:")
        max_index = output_data.index(max(output_data))  # 最大值的索引
        result.append(max_index)
        # print(output_data)
t = time.time() - t1
print(result)
print("总共费时：" + str(t) + "秒")


def np_count(nparray, x):
    i = 0
    for n in nparray:
        if n == x:
            i += 1
    return i


print("总数：" + str(len(result)) + "\n" + "正确率：" + str(np_count(result, label) / (len(result))))
print(np_count(result, label))
mytrue = [label] * len(result)
mypred = result
utils.plot_confusion_matrix2(mytrue, mypred, save_flg=True)
