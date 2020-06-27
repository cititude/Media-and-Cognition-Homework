#### 任务三

- ```sh train.sh```训练元模型，此后每隔10周期记录一次模型参数。我们训练好的模型保存在``` result/meta_model.pt```中。预计训练时间为10h。
- 可以在train.sh中使用```--meta_batch```以使用批训练，使用``` --use_mixup```以启用mixup数据增强。
- ```sh test.sh```可用于测试，其中使用参数```--finetune```时对元学习训练好的模型在测试数据训练集上进行训练。我们经此训练成的模型保存在```result/model.pt```中。使用参数```--only_test```对测试集数据进行测试,生成pred.json文件。