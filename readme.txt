1、下载casia数据集
训练集解压缩到raw\casia-pot\Pot1.0Train
测试集解压缩到raw\casia-pot\Pot1.0Test
.\tools\pot2txt 为casia pot文件转换为txt文件，已经编译linux和windows版本

2、执行.\make_data.py
linux下执行需要首先设置pot2txt的执行权限
chmod 755 ./pot2txt
在./data/casia-sample目录下生成样本数据

3、执行.\casia_v04_cnn.py
需要首先安装配置pytorch环境
脚本在./model/casia1_v04_cnn目录下存储模型以及log文件

