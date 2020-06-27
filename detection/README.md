### 任务四

- ```sh train.sh```对模型进行训练，生成pred_stage1.json文件。该文件包含所预测的预测框，但对预测类别没有保证。再执行 ```python filter.py --infile pred_stage1.json```后生成最终的pred.json
- ```python show.py```可以输出带有检测框的预测图片。
