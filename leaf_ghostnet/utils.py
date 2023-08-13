#!/usr/bin/python
# -*- coding:utf8 -*-
# import skimage
# import skimage.io
# import skimage.transform
import os

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

from sklearn.metrics import confusion_matrix

# freeze参考来源http://blog.csdn.net/lujiandong1/article/details/53385092

labels_name = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}


def freeze_graph(model_folder):
    # 读入checkpoint，并获取路径
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # 设置冻结模型保存的位置
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    # freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
    # 输出结点可以看我们模型的定义
    # 只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
    # 所以,output_node_names必须根据不同的网络进行修改（指定节点以前的都会保存下来）
    output_node_names = "prediction/predicted"

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # 这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
    # 相当于将参数已经固化在了图当中
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index] and "Switch" not in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        # We use a built-in TF helper to export variables to constant 将参数变成常量写入模型
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem  保存带参数的模型
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


# returns image of shape [224, 224, 3]
# [height, width, depth]
# def load_image(path):
#     # load image
#     img = skimage.io.imread(path)
#     img = img / 255.0
#     assert (0 <= img).all() and (img <= 1.0).all()
#     # print "Original Image Shape: ", img.shape
#     # we crop image from center
#     short_edge = min(img.shape[:2])
#     yy = int((img.shape[0] - short_edge) / 2)
#     xx = int((img.shape[1] - short_edge) / 2)
#     crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
#     # resize to 224, 224
#     resized_img = skimage.transform.resize(crop_img, (224, 224))
#     return resized_img

def load_graph(frozen_graph_filename):
    # We parse the graph_def file 解析图文件
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # We load the graph_def in the default graph  加载图文件为默认图
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


# sortmax 结果转 onehot
def props_to_onehot(props):
    if isinstance(props, list):  # 判断对象是否是已知类型
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b


def draw_matrix_acc(y_pred, y_true):
    tick_marks = np.array(range(len(labels_name))) + 0.5
    ind_array = np.arange(len(labels_name))
    cm = confusion_matrix(y_true, y_pred)  # 求解confusion matrix #需要对比类别数，不是onehot,而且是一个一维数组

    # print cm
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print cm_normalized
    f = plt.figure(figsize=(12, 8), dpi=20)

    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if (c > 0.01):
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='senet_tenbirds')
    # show confusion matrix
    plt.show()
    time.sleep(5)
    print("clear figure")
    f.clear()
    plt.close()


def plot_confusion_matrix(cm, title):  # 绘制混淆矩阵cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  #
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./logs/senetadm.png', format='png')  # 存储图片


def plot_confusion_matrix2(y_true, y_pred, title="Confusion matrix",
                           cmap=plt.cm.Blues, save_flg=False):
    myPath = './picture'
    if not os.path.exists(myPath):
        os.mkdir(myPath)
    classes = [str(i) for i in range(11)]  # 代表有11类
    labels = range(11)  # 数据集的标签类别，跟上面I对应，含首不含尾0-6
    cm = np.around(confusion_matrix(y_true, y_pred, labels=labels, normalize='true'), decimals=2)  # 获取混淆矩阵
    plt.figure(figsize=(14, 12))  # 定义窗口大小
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 显示混淆矩阵，插值方式，主题
    plt.title(title, fontsize=40)  # 标题，字体大小
    plt.colorbar()   # 显示颜色条
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)  # 覆盖坐标轴值，这里就是使用class覆盖数值
    plt.yticks(tick_marks, classes, fontsize=20)
    print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.  # 混淆矩阵最大值的一半
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):  # 这里取矩阵值和颜色，画出来
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=30)  # 设置标签
    plt.xlabel('Predicted label', fontsize=30)
    if save_flg:  # 保存
        plt.savefig(myPath + "/confusion_matrix.png")
