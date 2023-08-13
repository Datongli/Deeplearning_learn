import os
import subprocess
import librosa


# 查看采样率
def sortrate(path):
    input_path = path
    rate16 = 0
    rate44 = 0
    for file in os.listdir(input_path):  # 这里遍历类中的wav
        file_path = os.path.join(input_path, file)
        for filed in os.listdir(file_path):
            file1 = os.path.join(file_path, filed)
            if file1.endswith('wav'):
                _, lr = librosa.load(file1, sr=None)
                if lr == 16000:
                    rate16 += 1
                elif lr == 44100:
                    rate44 += 1
                else:
                    print(file1 + '采样率:' + str(lr))
    print('采样率为16000的有' + str(rate16) + '个，采样率为44100的有' + str(rate44) + '个')


# 改采样率
def relr(path):
    input_path = path
    input_path2 = path + '2'
    if not os.path.exists(input_path2):
        os.makedirs(input_path2)
    for file in os.listdir(input_path):  # 这里遍历类中的wav
        file_path = os.path.join(input_path, file)
        file_path2 = os.path.join(input_path2, file)
        if not os.path.exists(file_path2):
            os.makedirs(file_path2)
        for filed in os.listdir(file_path):
            if filed.endswith('wav'):
                file1 = file_path + '\\' + filed
                file2 = file_path2 + '\\' + filed
                cmd = 'ffmpeg -i ' + file1 + ' -ar 16000 ' + file2  # ffmpeg -i 输入文件 -ar 采样率  输出文件
                subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    mypath = r"F:\程序\leaf-ghostnet\datahard2"
    sortrate(mypath)
    # relr(mypath)
