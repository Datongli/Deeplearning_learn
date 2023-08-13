import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
from ghost_model import GhostNet
from math import pi
from imblearn.ensemble import EasyEnsembleClassifier


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# assert tf.__version__.startswith('2.')


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)


def _argment_helper(image):
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    # image = tf.image.resize(image, [227, 227])
    image = tf.math.divide(image, tf.constant(255.0))
    return image


def parse_fn(example_proto):
    """Parse TFExample records and perform simple data augmentation."""
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    # image = tf.image.decode_jpeg(parsed['image_raw'], 3)
    image = tf.io.decode_raw(parsed['data'], tf.float32)
    # image = tf.reshape(image, [20, 1200])
    # image = _argment_helper(image)
    label = tf.cast(parsed['label'], tf.int64)
    y = tf.one_hot(label, 5)

    return image, label


def input_fn(name):
    dataset = tf.data.TFRecordDataset(name + '.tfrecords')
    dataset = dataset.repeat()   #循环训练，去掉的话，只训练一遍数据
    dataset = dataset.shuffle(buffer_size=2048)  # 意义？？？
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=2)
    dataset = dataset.batch(batch_size = batch_size)
    return dataset


# 定义损失函数
def custom_loss(y_actual, y_pred):  #用于处理数据不均衡，focus loss
    logits = y_pred
    labels = y_actual
    # 1-ratio of each class
    alpha = [[0.92], [0.93], [0.89], [0.92], [0.90], [0.91], [0.88], [0.93], [0.92], [0.90], [0.91]]
    epsilon = 1.e-7
    gamma = 2

    logits = tf.reshape(logits, [batch_size, logits.shape[2]])
    labels = tf.reshape(labels, [batch_size])
    # (Class ,1)
    alpha = tf.constant(alpha, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.int32)
    logits = tf.cast(logits, tf.float32)
    # (N,Class) > N*Class
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]  # [batch_size * n_class]
    # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
    labels_shift = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
    # labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
    # (N*Class,) > (N,)
    prob = tf.gather(softmax, labels_shift)
    # 预防预测概率值为0的情况  ; (N,)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    # (Class ,1) > (N,)
    alpha_choice = tf.gather(alpha, labels)
    # (N,) > (N,)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    # (N,) > 1
    loss = -tf.reduce_mean(tf.multiply(weight, tf.math.log(prob)))
    return loss

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# FRAME_LEN = 1200
# SEQUENCE_LEN = 20
batch_size = 32  # 64,0.9-0.8
trainsize = 3581
testsize = 965
model = GhostNet(2)
# model = tf.keras.models.load_model('./model_saved')  # 这里用于断点续训

# w0 = trainsize/(2*3401)
# w1 = trainsize/(2*180)
# class_weights = {0:0.52646,1:9.4722}


class MyCosDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps

    def __call__(self, step):
        return self.initial_learning_rate * (1 + tf.math.cos(step * pi /self.decay_steps))/2

lr_schedule = MyCosDecay(
    initial_learning_rate = 0.0001,
    decay_steps=36)

model.compile(
    loss='sparse_categorical_crossentropy', #内置的loss函数
    #loss=custom_loss,  #focus loss
    #optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, decay=0.0001), # 'sparse_categorical_crossentropy'
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,decay = 0.001),#, decay=0.01
    #optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule),#, decay=0.01
    #optimizer=tf.keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.01),
    #metrics=['accuracy']
    metrics=[auc]
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)  # 模型编译


dataset = input_fn('fold1_train')
testset = input_fn('fold1_val')
# predictset = input_fn('predict')
myList = []
myList2 = []
# for each, each2 in dataset.take(1):
#     print(each.shape)
#     print(each2.shape)


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, mode='min')
# checkpoint_path = "../save_model/train-{epoch:04d}.ckpt"
# callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=5)
# model.save_weights(checkpoint_path.format(epoch=0))  11407 3116
# history = model.fit(dataset, epochs=10, steps_per_epoch=100, callbacks=[callback])

# # 测试每一张图
# for each, each2 in dataset.take(1):
#     print(each2[0])
#     plt.imshow(each[0][:, :, ::-1])
#     plt.show()

# model.build((None, 224, 224, 3))
model.summary_model()

history = model.fit(dataset, epochs=50, steps_per_epoch=trainsize // batch_size, validation_data=testset,
                    validation_steps=testsize // batch_size, callbacks=None, shuffle=True)

scores = []
# 用于测试
for each, each2 in testset.take(testsize // batch_size):
    predict_classes = model.predict(each, batch_size=batch_size)
    for predict_class in predict_classes:
        scores.append(predict_class)
        myList.append(np.argmax(predict_class[0]))
    for each3 in each2:
        myList2.append(each3.numpy())
print(myList2.__len__())
print(myList.__len__())
utils.plot_confusion_matrix2(myList2, myList, save_flg=True)  # 调用混淆矩阵8996 2570 1291

for item in scores:
    content = str(item) + "\n"
    with open('scores_fold_1.txt','a') as f:
        f.write(content)
for item2 in myList2:
    content2 = str(item2) + "\n"
    with open('labels_fold_1.txt', 'a') as f:
        f.write(content2)

# 保存模型 10282, 2527
modelPath = './model_saved'
if not os.path.exists(modelPath):
    os.makedirs(modelPath)
tf.saved_model.save(model, modelPath)
model = tf.saved_model.load(modelPath)
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, 15360])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter = tf.lite.TFLiteConverter.from_saved_model(modelPath)
# converter.experimental_new_converter = True
tflite_model = converter.convert()
with open('chirpModel.tflite', 'wb') as f:
    f.write(tflite_model)


def plot_metrics(history):
    metrics = ['loss', 'accuracy']
    # metrics = ['loss', 'sparse_categorical_accuracy']
    myPath = './picture'
    if not os.path.exists(myPath):
        os.mkdir(myPath)
    for n, metric in enumerate(metrics):
        name = metric
        # plt.subplot(1, 2, n + 1)
        # print(history.history)
        plt.figure()
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color="coral", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title('train ' + name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])  # 这里ylim设置下限为0
            plt.legend(["train", "test"], loc="upper right")
            plt.savefig(myPath + '/train_loss.png')
        else:
            plt.legend(["train", "test"], loc="lower right")
            plt.savefig(myPath + '/train_accuracy.png')
        # plt.show()


plot_metrics(history)
