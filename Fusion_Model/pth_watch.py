import torch


# 加载.pth文件
# checkpoint = torch.load(r"C:\Users\ldt20\Desktop\model_Coswara（原始+增强）谱图+音频 并联网络最好的权重parallel_model网络.pth")
checkpoint = torch.load(r"C:\Users\ldt20\Desktop\训练权重保存\23.5.4后的\resnet+tcnn并联（预训练后）.pth")

# # 查看.pth文件中的内容
print(checkpoint.keys())  # 打印.pth文件中的键列表
print(type(checkpoint.keys()))
print("fc.weight")
print(len(checkpoint['fc.weight']))
print(checkpoint['fc.weight'])

print("fc.bias")
print(len(checkpoint['fc.bias']))
print(checkpoint["fc.bias"])

print("fc_2.weight")
print(checkpoint['fc_2.weight'])
print(len(checkpoint['fc_2.weight']))

print("fc_2.bias")
print(len(checkpoint['fc_2.bias']))
print(checkpoint["fc_2.bias"])

# # 查看.pth文件的完整内容
# print(checkpoint)
#
# print(checkpoint['fc.weight'])
# print(len(checkpoint['fc.weight']))

# 只查看pth文件中保存的模型参数，不包含batch等信息
# model_state_dict = checkpoint['model_state_dict']
# print(model_state_dict)
