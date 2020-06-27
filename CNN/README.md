### 总体方案

- 我们采用sedensenet模型，对训练集数据训练了300个周期，得到最终的模型。此外，我们事先从数据中hold out了1000条数据用作验证集，并根据验证集结果选出最佳模型作为最终结果。

### 细节

- 我们在不破坏语义（例如箭头方向翻转或警示牌颜色变化）的前提下对图像进行了数据增强，使用了RandomRotation和ColorJitter，见datasetsets.py #29.
- 我们也尝试了resnet和densenet结构训练了20周期，发现验证集准确率sedensenet>densenet>resnet，这证实了densenet和注意力机制在本问题上具有一定的优势，故最终使用了sedensenet结构。在```model_name```参数可以修改为densenet和resnet以运行不同的模型。

### 代码结构

| 文件名      | 内容                                                         |
| ----------- | ------------------------------------------------------------ |
| models.py   | 定义了模型SEDensenet的结构                                   |
| datasets.py | 实现了数据集的读入                                           |
| utils.py    | 实现了一些工具函数，如日志函数，计数器函数和种子初始化       |
| main.py     | 主函数部分。其中实现了train,val和test 3个基本函数，用于训练+验证+测试。 |
| cam.py      | 定义了CAM(class activation map)，实现了分类网络的热力图可视化 |

## 运行方法
- ```sh train.sh```对模型进行训练，注意修改```--data_path +分类数据路径 ```。默认使用``` fix_seed```可以复现模型。共用时约20h。
- ``` sh test.sh ```对训练好的模型进行测试集上的测试，并生成```pred.json```文件，修改参数``` pred_file```可以修改生成文件的位置。
- ``` sh visualize.sh```可以创建heatmaps文件夹，并在其中生成交通标志分类热力图。
