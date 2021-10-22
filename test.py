import paddlehub as hub
import cv2
import os
from paddle.inference import Config
from paddle.inference import create_predictor
module = hub.Module(directory='DriverStatusRecognition')
data_path = "../data/imgs/test"
images = []
labels = []

# 读取测试集图片和标签
with open("./test_list.txt", 'r') as readfile:
    for line in readfile.readlines():
        components = line.split(" ")
        file_path = components[0]
        label = components[1]
        images.append(file_path)
        labels.append(label)
if(len(images)!=len(labels)):
    print("Error, the num of images doesn't match the num of label")
    readfile.close()
    exit
# 开始测试
results = module.predict(images=images)
# 统计测试总数和正确率
total = 0
correct = 0
with open("./result.csv", 'w') as writefile:
    for i in range(len(results)):
        if(results[i][0]["category_id"]==int(labels[i])):
            correct = correct+1
            writefile.write(images[i]+","+str(results[i][0]["category_id"])+","+labels[i]+",T")
        else:
            writefile.write(images[i]+","+str(results[i][0]["category_id"])+","+labels[i]+",F")
        total = total+1
# 输出正确率
print("acc:"+str(correct/total))
