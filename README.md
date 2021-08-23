CRNN
======================================

本人环境
----------
* pytorch=1.6.0
* cuda=10.2
* cudnn=0.8

环境依赖
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding),直接编译放在当前目录下就行

* lmdb==0.97
* numpy==1.17.2
* Pillow==6.1.0
* six==1.12.0
* torch>=1.2.0
* torchvision>=0.4.0

数据合成
----------
* ./make_data,用于生成数据,也可以用其他的
* ./make_data/bg 存放背景图片
* ./make_data/fonts 存放字体
* ./make_data/create_str_num.py 生成样本，默认是生成数字和英文,可以添加中文语料
* ./make_data/to_images_list.py 生成的包含样本路径和标签信息的txt文件,格式为：图片路径+空格+标签
* 生成的格式不一定固定,可以按个人习惯调整代码

数据标注
----------
* 生成包含实际图片样本路径的txt文件
* 将图片样本中包含的字符标注到txt中对应路径后面
* 最终格式：txt中每一行代表一个数据,即图片路径+空格+标签

lmdb生成
----------
* ./create_dataset 简单修改对应路径直接生成训练和验证集

训练
----------
* ./train 调用生成的lmdb数据集训练模型
* 模型文件存储在expr中，用于测试和生成wts和onnx文件

生成wts或onnx
----------
* ./genwts.py 生成wts文件，用于[静态输入crnn算法](http://git.yuntongxun.com/liwei11/trt_crnn_static_cpp.git)
* ./make_onnx.py 生成onnx文件，用于[动态输入crnn算法](http://git.yuntongxun.com/liwei11/trt_crnn_dynamic_cpp.git)

注意事项
----------
* 使用中文语料，注意字符编码问题
* warp-ctc编译时，需要注意与pytorch，cuda，cudnn的版本兼容问题
