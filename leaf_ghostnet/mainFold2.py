import tensorflow as tf
import numpy as np
from ghost_model import GhostNet
from sklearn.model_selection import KFold

batch_size = 20
trainsize = 9727
testsize = 4175
def parse_fn(example_proto):
    """Parse TFExample records and perform simple data augmentation."""
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)

    image = tf.io.decode_raw(parsed['data'], tf.float32)

    label = tf.cast(parsed['label'], tf.int64)
    return image, label

def input_data(name):
    print('加载tfrecord')
    dataset = tf.data.TFRecordDataset(name)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=2)
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset

traindata = input_data('trainnew.tfrecords')
testdata = input_data('valnew.tfrecords')

trainlabel = []
traininputs = []
for each, each2 in traindata.take(trainsize):
    each = each.numpy()
    each = each[0]
    traininputs.append(each)
    #print(inputs)
    each2 = each2.numpy()
    each2 = each2[0]
    trainlabel.append(each2)
    #print(label)

testlabel = []
testinputs = []
for eachh, eachh2 in testdata.take(testsize):
    eachh = eachh.numpy()
    eachh = eachh[0]
    testinputs.append(eachh)

    eachh2 = eachh2.numpy()
    eachh2 = eachh2[0]
    testlabel.append(eachh2)

traininputs = np.array(traininputs)
testinputs = np.array(testinputs)
trainlabel = np.array(trainlabel)
testlabel = np.array(testlabel)
print('traininuts:',traininputs.shape)
print('testinputs:',testinputs.shape)
print('trainlabel:',trainlabel.shape)
print('testlabel:',testlabel.shape)
# Merge inputs and targets
inputs = np.concatenate((traininputs, testinputs), axis=0)
#inputs = np.expand_dims(inputs,axis=0)
targets = np.concatenate((trainlabel, testlabel), axis=0)
#targets = np.expand_dims(targets,axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold = []
loss_per_fold =[]
for train, test in kfold.split(inputs, targets):
    # Define the model architecture
    model = GhostNet(2)

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',  # 内置的loss函数
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=0.001),  # , decay=0.01
        # optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule),#, decay=0.01
        metrics=['accuracy']
    )  # 模型编译

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(inputs[train], targets[train], epochs=1, steps_per_epoch=trainsize // batch_size,
                        validation_steps=testsize // batch_size, callbacks=None, shuffle=True)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

print(acc_per_fold)
print(loss_per_fold)
[48.81763458251953, 87.73055076599121, 87.24939823150635, 85.28468608856201, 85.72574257850647]
[0.693146824836731, 1.6086037158966064, 1.7627203464508057, 1.8603756427764893, 1.891952395439148]