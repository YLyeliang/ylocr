# OCR数据集介绍

## 中文数据集

### ICDAR2019-LSVT

总共45W张中文街景图像，包括**3W 完全标注**训练数据(**文本坐标 + 文本内容**)，2W测试集。和40W只标注了文本内容的图像。

Data sources: [链接](https://ai.baidu.com/broad/introduction?dataset=lsvt)

### ICDAR2017-RCTW-17

1W2+ 图片，大部分是户外手机拍摄，部分为截图。包括多种场景：街景、海报、菜单、室内和应用截图。

### Chinese Street View Text Recognition

总共29W，其中21W作为训练集，8W为测试集。为中国街景，并且是**已经切割好的文本区域**，可以直接用于识别任务，高为48.

下载链接：[aistudio.baidu.com](https://aistudio.baidu.com/aistudio/datasetdetail/8429)

### Chinese Document Text Recognition

- 总共364W，按照99：1的比例分成训练和验证集。

- 使用中文语料库(新闻+经典汉字),数据经过字体、大小、灰度值、模糊、透视、拉伸等变换随机生成

- 5990个字符，包含中文字符、英文字母、数字和标点符号(字符集):[github.com/YCG09](https://github.com/YCG09/chinese_ocr/blob/master/train/char_std_5990.txt)
- 每个样本固定10个字符，字符是从句子中随机截断。
- 分辨率为280x32

链接:[百度云](https://pan.baidu.com/s/1QkI7kjah8SPHwOQ40rS1Pw)

### ICDAR2019-ArT

包含10166图片，其中5603为训练，4563为测试集。包含三部分：total text, scut-ctw1500, Baidu curved scene text，包含不同形状的文本，比如水平，多方向的和弧形的。

链接：[ai.baidu.com](https://ai.baidu.com/broad/download?dataset=art)



## 英文数据集
