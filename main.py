import os
import paddlex as pdx
import paddlehub as hub
import paddle.fluid as fluid
import numpy as np
from paddlex.cls import transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 加载数据
def load_data(data_base_path):
    datas = []
    for i in range(10):
        data_path = data_base_path + '/imgs/train/c{}/'.format(i)
        for im in os.listdir(data_path):
            pt = os.path.join(data_base_path+'/imgs/train/c{}/'.format(i), im)
            line = '{} {}'.format(pt, i)
            datas.append(line)

    np.random.seed(10)
    np.random.shuffle(datas)

    total_num = len(datas)
    train_num = int(0.8*total_num)
    test_num = int(0.1*total_num)
    valid_num = total_num - train_num - test_num

    print('train:', train_num)
    print('valid:', valid_num)
    print('test:', test_num)

    with open('train_list.txt', 'w') as f:
        for v in datas[:train_num]:
            f.write(v+'\n')

    with open('test_list.txt', 'w') as f:
        for v in datas[-test_num:]:
            f.write(v+'\n')

    with open('val_list.txt', 'w') as f:
        for v in datas[train_num:-test_num]:
            f.write(v+'\n')

    with open('labels.txt', 'w') as f:
        for i in range(10):
            f.write('ch{}\n'.format(i))

    train_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.RandomCrop(crop_size=224),
    transforms.RandomDistort(),
    transforms.Normalize()
    ])
    eval_transforms = transforms.Compose([
        transforms.ResizeByShort(short_size=256),
        transforms.CenterCrop(crop_size=224),
        transforms.Normalize()
    ])
    train_dataset = pdx.datasets.ImageNet(
        data_dir='',
        file_list='train_list.txt',
        label_list='labels.txt',
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.ImageNet(
        data_dir='',
        file_list='val_list.txt',
        label_list='labels.txt',
        transforms=eval_transforms)

    num_classes = len(train_dataset.labels)
    print(num_classes)
    return train_dataset, eval_dataset, num_classes

def model_train(train_dataset, eval_dataset, num_classes):
    model = pdx.cls.MobileNetV3_small_ssld(num_classes=num_classes)
    model.train(num_epochs=20,
            train_dataset=train_dataset,
            train_batch_size=32,
            log_interval_steps=20,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[1],
            save_interval_epochs=1,
            learning_rate=0.01,
            save_dir='output/mobilenetv3')

def model_eval(eval_dataset):
    save_dir = 'output/mobilenetv3/best_model'
    model = pdx.load_model(save_dir)
    model.evaluate(eval_dataset, batch_size=1, epoch_id=None, return_details=False)

if __name__ == "__main__":
    train_dataset, eval_dataset, num_classes=load_data("../data")
    model_train(train_dataset, eval_dataset, num_classes)
    model_eval(eval_dataset)

