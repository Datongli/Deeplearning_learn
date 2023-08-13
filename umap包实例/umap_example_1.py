"""
该文件用于实验我们现在手中有的数据
使用umap包绘图
"""

import os
from skimage.io import imread
import numpy as np
import umap
import matplotlib.pyplot as plt
import datetime
from umap import UMAP


# 读取两个文件夹中的图像数据
def load_data(data_path, species):
    data = []
    labels = []
    for label, folder in enumerate(species):
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.png'):
                image_path = os.path.join(folder_path, file)
                image = imread(image_path)
                data.append(image.flatten())
                labels.append(label)
    return np.array(data), np.array(labels)

# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# 读取数据
data, labels = load_data(r"D:\学习\大创\data\训练数据集\data\mnist_1", species=['0','1','2','3','4','5','6','7','8','9'])
# data, labels = load_data(r"C:\Users\ldt20\Desktop\negative_positive", species=['negative', 'positive'])

print(data)
print(type(data))
print(data.shape)

print('over1')
# 将数据降维到2维并绘制分布图
# reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1, n_components=2)
# embedding = reducer.fit_transform(data)
# plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='coolwarm', s=10)
# plt.title('UMAP projection of image data')
# plt.savefig(os.path.join(r"C:\Users\ldt20\Desktop\umap绘图", "猫狗绘图{}".format(nowTime)))
# plt.show()



reducer = umap.UMAP(random_state=42)
reducer.fit(data)

UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)

embedding = reducer.transform(data)
assert(np.all(embedding == reducer.embedding_))
print(embedding)
print(embedding.shape)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=15)
plt.show()

print('over2')
