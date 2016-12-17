# 《统计学习方法》笔记与算法实现

## 说明
李航《统计学习方法》笔记与算法的 Python 实现。测试样例是例题数据，学习笔记主要在[本人博客](http://121.42.47.99/yuenshome/wordpress)的[《统计学习方法》](http://121.42.47.99/yuenshome/wordpress/?cat=202)目录下，也可点击下方目录的章节名阅读笔记。笔记大部分为原书的内容摘抄。

算法主要基于 Python 语言进行实现，为了原汁原味地用代码将算法描述，所有没有使用第三方的线性代数运算库（如 Numpy 或 Pandas 等）。代码注释尽可能完善完整，力求描述准确到位。如有错误，还望指出（[→ 点击这里提问题吧 ←](https://github.com/ysh329/statistical-learning-methods-note/issues)），不胜感激！

## 目录

* 第 1 章 统计学习方法概论
* 第 2 章 [感知机](./chapter_2_perceptron/) [\[感知机代码-原始形式\]](./chapter_2_perceptron/Perceptron.py) [\[感知机代码-对偶形式\]](./chapter_2_perceptron/Dual-form_Perceptron.py)  
* 第 3 章 [k近邻算法](./chapter_3_kNN/) [\[k近邻代码\]](./chapter_3_kNN/kNN.py) [\[kd树简化版代码\]](./chapter_3_kNN/Simple-kd-Tree.py) [\[kd树完整版代码\]](./chapter_3_kNN/kd-Tree.py) [\[错误kd树代码\]](./chapter_3_kNN/WrongKDTreeCodeDemo.py)
* 第 4 章 [朴素贝叶斯法](./chapter_4_NaiveBayes/)
* 第 5 章 决策树
* 第 6 章 逻辑斯谛回归与最大熵模型
* 第 7 章 支持向量机
* 第 8 章 [提升方法](./chapter_8_boosting/) [\[AdaBoost代码\]](./chapter_8_boosting/AdaBoost.py)
* 第 9 章 EM算法及其推广
* 第 10 章 隐马尔科夫模型
* 第 11 章 条件随机场
* 第 12 章 统计学习方法总结