import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import time
import utils

# Load TFLite model and allocate tensors.
tflite_model = tf.lite.Interpreter(model_path="chirpfocal.tflite")
tflite_model.allocate_tensors()

# Get input and output tensors.
input_details = tflite_model.get_input_details()
output_details = tflite_model.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # 输入随机数
# print(input_data)  # (1, 224, 224, 3)

path = "test_picture/"
contents = os.listdir(path)
result = []
t1 = time.time()
for each in contents:
    # print(contents)
    # image = cv.imread(path + each)[np.newaxis, :]
    image = cv.imdecode(np.fromfile(path + each, dtype=np.uint8), cv.IMREAD_COLOR)[np.newaxis, :]
    input_data = np.array(image, dtype=np.float32) / 255.0
    # print(image)
    tflite_model.set_tensor(input_details[0]['index'], input_data)
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


print("总数：" + str(len(result)) + "\n" + "正确率：" + str(np_count(result, 0) / (len(result))))
print(np_count(result, 0))
mytrue = [0] * len(result)
mypred = result
utils.plot_confusion_matrix2(mytrue, mypred, save_flg=True)
