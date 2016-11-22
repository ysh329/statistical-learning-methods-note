# -*- coding: utf-8 -*-
################################### PART0 DESCRIPTION #################################
# Filename: Dual-form_Perceptron.py
# Description:
#

# E-mail: ysh329@sina.com
# Create: 2016-11-16 21:55:59
# Last:
__author__ = 'yuens'


################################### PART1 IMPORT ######################################
import random

################################### PART2 CLASS && FUNCTION ###########################

def readDataFrom(path, hasHeader=True):
    '''
    读取路径为path的文件，默认第一行为表头文件(hasHeader=True)，
    否则需要设置第一行不包含表头文件(hasHeader=False)。
    :param path: 读取数据的路径
    :param hasHeader: 数据文件是否有表头
    :return: 返回数据id、特征、标签
    '''
    with open(path, 'r') as f:
        rawData = map(lambda line:\
                          line.strip().split(" "),\
                      f.readlines())
        if hasHeader:
            header = rawData[0]
            print("header:{0}".format(header))
            cleanData = map(lambda recordList: \
                                map(int, recordList), \
                            rawData[1:])
        else:
            print("header:None")
            cleanData = map(lambda recordList: \
                                map(int, recordList), \
                            rawData)
        idList = map(lambda r: r[0], cleanData)
        xList = map(lambda r: r[1:len(r)-1], cleanData)
        yList = map(lambda r: r[-1], cleanData)
        return idList, xList, yList


class DualFormPerceptron(object):
    '''
    对偶形式-感知器类，基于随机梯度下降法训练感知器算法，遇到分类分类错误样本进行参数更新。
    '''
    def __init__(self, sampleNum, featureNum, learningRate=10E-4):
        '''
        初始化感知器。
        :param sampleNum: 训练集样本数目
        :param featureNum: 每个样本的特征数
        :param learningRate: 随机梯度下降算法中的参数学习率
        '''
        # 初始化超参数
        self.sampleNum = sampleNum
        self.featureNum = featureNum
        self.eta = learningRate

        # 随机初始化参数
        self.alphaList = [map(lambda i: \
                                  0.0, #random.random(),
                              xrange(self.sampleNum))\
                          ]
        self.bList = [0.0] #[random.random()]

    def constructGramMatrix(self, xList):
        '''
        构造 Gram 矩阵。该矩阵中的每个元素是每两个样本间对应特征的点积和。
        :param xList: 输入样本特征
        :return:
        '''
        self.gramMatrix = [[0 for col in xrange(self.sampleNum)] for row in xrange(self.sampleNum)]
        print self.gramMatrix
        for idx1 in xrange(self.sampleNum):
            for idx2 in xrange(self.sampleNum):
                if idx1 <= idx2:
                    innerProd = sum(\
                        map(lambda xx1, xx2:\
                                xx1 * xx2,\
                            xList[idx1], xList[idx2])\
                        )
                    self.gramMatrix[idx1][idx2] = self.gramMatrix[idx2][idx1] = innerProd

    def train(self, xList, yList, maxEpochNum):
        '''
        基于随机梯度下降算法对感知器参数更新，训练最大轮数为 maxEpoch ，
        当训练集完全预测正确则停止训练。
        :param xList: 训练集的特征数据
        :param yList: 训练集的真实类别
        :param maxEpochNum: 最大迭代次数，若某轮 epoch 预测完全正确则
        停止迭代。
        :return:
        '''
        # costList 和 misRateList 用于记录每轮 epoch 的损失函数
        # 值和错分率
        costList = []
        misRateList = []
        for epochIdx in xrange(maxEpochNum):
            print("======= epochIdx {0} =======".format(epochIdx))
            curEpochCostList = []
            for sampleIdxI in xrange(len(xList)):
                x = xList[sampleIdxI]
                yHat, sigma = self.predict(x,\
                                           xList,\
                                           yList,\
                                           useGramMatrix=True,\
                                           sampleIdxI=sampleIdxI,\
                                           iterIdx=None)
                cost = yList[sampleIdxI] * sigma
                curEpochCostList.append(cost)
                # 打印cost
                iterIdx = epochIdx * len(xList) + sampleIdxI
                print("== iterIdx:{0} ==".format(iterIdx))
                print("cost:{0}".format(cost))
                # 判断是否进行参数更新
                if cost <= 0:
                    nextAlpha = self.alphaList[epochIdx]
                    nextAlpha[sampleIdxI] += self.eta
                    #nextAlpha = self.alphaList[epochIdx][sampleIdxI] + self.eta
                    nextB = self.bList[epochIdx] + self.eta * yList[sampleIdxI]
                else:
                    nextAlpha = self.alphaList[-1]
                    nextB = self.bList[-1]
                self.alphaList.append(nextAlpha)
                self.bList.append(nextB)
            # 判断当前参数的预测性能
            curEpochPredictTupleList = map(lambda x:\
                                           self.predict(x, xList, yList),\
                                       xList)
            curEpochResultList = map(lambda idx, (yHat, sigma), y:\
                                         (idx, yHat, y, y == yHat, sigma),\
                                     xrange(len(curEpochPredictTupleList)), curEpochPredictTupleList, yList)
            curEpochCorrectNum = len(\
                filter(lambda (idx, yHat, y, isCorrectPredict, sigma):\
                           isCorrectPredict,\
                       curEpochResultList)\
                )
            misRate = 1.0 - float(curEpochCorrectNum) / len(curEpochResultList)
            print("misRate:{0}".format(misRate))
            misRateList.append(misRate)
            curEpochCost = sum(curEpochCostList)
            print("cost:{0}".format(curEpochCost))
            costList.append(curEpochCost)
            print
            if misRate == 0.0:
                break

        # 最终参数打包为字典
        parameterDict = dict()
        parameterDict['alpha'] = self.alphaList[-1]
        parameterDict['b'] = self.bList[-1]
        return parameterDict, costList, misRateList

    def predict(self, x, xList, yList, useGramMatrix=False, sampleIdxI=None, iterIdx=-1):
        '''
        预测输入样本 x 的所属类别。由于是对偶形式，所以也需要传入训练数据集的特征和标签，
        训练过程中，需要指定 userGramMatrix , sampleIdxI 参数，
        iterIdx 可以使用模型训练完成后使用某 iterIdx 次训练得到的模型参数进行预测，
        默认使用最后一次模型参数。
        :param x: 输入样本 x 的特征
        :param xList: 训练数据集的特征
        :param yList: 训练数据集的真实标签
        :param useGramMatrix: 是否使用 Gram 矩阵，训练时需指定
        :param sampleIdxI: 当前样本位于训练集的下标，训练时需指定
        :param iterIdx: 使用哪一次的模型参数预测，默认使用最后一次
        :return: 返回输入样本 x 的预测类别和 sigma
        '''
        # 是否使用 Gram 矩阵
        if useGramMatrix:
            sigma = sum(\
                    map(lambda sampleIdxJ:\
                            self.alphaList[iterIdx][sampleIdxJ] *\
                            yList[sampleIdxJ] *\
                            self.gramMatrix[sampleIdxI][sampleIdxJ],\
                        xrange(len(xList)))\
                    ) + self.bList[iterIdx]
        else:
            sigma = sum(\
                map(lambda sampleIdxJ:\
                        self.alphaList[iterIdx][sampleIdxJ] *\
                        yList[sampleIdxJ] *\
                        sum(\
                            map(lambda xx1, xx2:\
                                    xx1 * xx2,\
                                x, xList[sampleIdxJ])\
                            ),\
                    xrange(len(xList)))\
                ) + self.bList[iterIdx]

        yHat = self.sign(sigma)
        return yHat, sigma

    def sign(self, v):
        '''
        符号函数，传入参数 v 大于 0 则为返回 1 ，小于 0 返回 -1 ，
        等于 0 则返回 0 。
        :param v: 传入参数
        :return: 返回传入参数的正负性
        '''
        if v > 0.0:
            return 1
        elif v == 0.0:
            return 0
        else:
            return -1

    def plotChart(self, costList, misRateList, saveFigPath):
        '''
        绘制错分率和损失函数值随 epoch 变化的曲线。
        :param costList: 训练过程中每个epoch的损失函数列表
        :param misRateList: 训练过程中每个epoch的错分率列表
        :return:
        '''
        # 导入绘图库
        import matplotlib.pyplot as plt
        # 新建画布
        plt.figure('Perceptron Cost and Mis-classification Rate', figsize=(8, 9))
        # 设定两个子图和位置关系
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        # 选择子图1并绘制损失函数值折线图及相关坐标轴
        plt.sca(ax1)
        plt.plot(xrange(1, len(costList) + 1), costList, '--b*')
        plt.xlabel('Epoch No.')
        plt.ylabel('Cost')
        plt.title('Plot of Cost Function')
        plt.grid()
        ax1.legend(u"Cost", loc='best')

        # 选择子图2并绘制错分率折线图及相关坐标轴
        plt.sca(ax2)
        plt.plot(xrange(1, len(misRateList) + 1), misRateList, '-r*')
        plt.xlabel('Epoch No.')
        plt.ylabel('Mis-classification Rate')
        plt.title('Plot of Mis-classification Rate')
        plt.grid()
        ax2.legend(u'Mis-classification Rate', loc='best')

        # 显示图像并打印和保存
        # 需要先保存再绘图否则相当于新建了一张新空白图像然后保存
        plt.savefig(saveFigPath)
        plt.show()

################################### PART3 TEST ########################################
# 例子
if __name__ == "__main__":
    # 参数初始化
    dataPath = "./input"
    learningRate = 10E-3
    maxEpochNum = 10000
    saveFigPath = "./DualFormPerceptronPlot.png"

    # 读取训练数据
    idList, xList, yList = readDataFrom(path=dataPath,\
                                        hasHeader=True)
    print("idList:{0}".format(idList))
    print("xList:{0}".format(xList))
    print("yList:{0}".format(yList))

    # 对偶形式感知机类的实例化
    dfp = DualFormPerceptron(sampleNum=len(xList),\
                             featureNum=len(xList[0]),\
                             learningRate=learningRate)
    # 基于训练集构造 Gram 矩阵
    dfp.constructGramMatrix(xList=xList)
    # 训练模型
    parameterDict, costList, misRateList = dfp.train(xList=xList,\
                                                     yList=yList,\
                                                     maxEpochNum=maxEpochNum)
    # 绘制模型的损失函数及错分率性能图像
    dfp.plotChart(costList=costList,\
                  misRateList=misRateList,\
                  saveFigPath=saveFigPath)