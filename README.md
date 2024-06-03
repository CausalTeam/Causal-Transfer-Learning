# Causal-Transfer-Learning

## 简介

Causal-Transfer-Learning 是一种因果迁移学习方法。其通过生成代表不同人群的数据集，在样本量足够的数据集中训练多层感知机分类器模型，并评估其在样本量稀疏的数据集上的表现。

假设有大量（10000个）50岁左右的群体与已有少量（5个）30岁左右群体的样本，通过分析吸烟对于血压在两个人群的影响因子，将大量50岁左右的群体样本迁移至30岁左右群体，以丰富样本容量，从而更加准确地预测30岁左右群体是否患有心脏病。

## 环境安装

推荐使用conda安装虚拟环境，推荐使用ubuntu系统，在命令行中运行：
```bash
conda create -n ctl python=3.9
conda activate ctl
pip install scikit-learn
```

## 运行
运行：
```bash
python main.py
```

