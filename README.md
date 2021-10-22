使用说明：
目录结构：
```
.
├── data 训练与测试数据
│   ├── driver_imgs_list.csv  
│   ├── imgs
│   └── sample_submission.csv
│  源代码
└── paddle_solution
    ├── convert.sh
    ├── DriverStatusRecognition
    ├── evaluate.py
    ├── inference_model
    ├── labels.txt
    ├── main.py
    ├── output
    ├── outputs
    ├── README.md
    ├── test_list.txt
    ├── test.py
    ├── train_list.txt
    └── val_list.txt
```
使用main.py进行训练，通过conver.sh脚本进行模型转换，得到可部署模型，保存在outputs/下；
部署时使用gzip和tar进行解压，然后通过`hub install DriverStatusRecognition`进行部署安装，然后运行test.py进行模型测试；不需要使用时用`hub uninstall`卸载

数据集说明：
```
    'c0': 'normal driving',

    'c1': 'texting-right',
    
    'c2': 'talking on the phone-right',
    
    'c3': 'texting-left',
    
    'c4': 'talking on the phone-left',
    
    'c5': 'operating the radio',
    
    'c6': 'drinking',
    
    'c7': 'reaching behind',
    
    'c8': 'hair and makeup',
    
    'c9': 'talking to passenger'
```