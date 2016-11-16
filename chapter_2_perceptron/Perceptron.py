# -*- coding: utf-8 -*-
################################### PART0 DESCRIPTION #################################
# Filename: Perceptron.py
# Description:
#

# E-mail: ysh329@sina.com
# Create: 2016-11-15 09:28:18
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


class Perceptron(object):
    '''
    感知器的类，基于随机梯度下降法训练感知器算法。
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
        self.wList = [map(lambda wIdx:\
                             random.random(),\
                         xrange(featureNum))]
        self.bList = [random.random()]

    def train(self, xList, yList, maxEpoch):
        '''
        基于随机梯度下降算法对感知器参数更新，训练最大轮数为 maxEpoch ，
        当训练集完全预测正确则停止训练。
        :param xList: 训练集的特征数据
        :param yList: 训练集的真实类别
        :param maxEpoch: 最大迭代次数，若某轮 epoch 预测完全正确则
        停止迭代。
        :return:
        '''
        # costList 和 misRateList 用于记录每轮 epoch 的损失函数
        # 值和错分率
        costList = []
        misRateList = []
        for epochIdx in xrange(maxEpoch):
            # 记录当前这一轮的预测类别，
            # 在之后判断本轮是否全部预测正确
            # 全部正确则停止迭代
            curEpochYHatList = []
            cost = 0.0
            errNum = 0.0
            print("======= epochIdx {0} =======".format(epochIdx))
            for sampleIdx in xrange(len(xList)):
                # 初始参数
                iterIdx = epochIdx * len(xList) + sampleIdx
                x = xList[sampleIdx]
                y = yList[sampleIdx]
                yHat, summation = self.predict(x)
                curEpochYHatList.append(yHat)
                cost += -summation * y
                # 打印中间结果
                print("== iterIdx:{0} ==".format(iterIdx))
                print("x:{0}".format(x))
                print("y:{0}".format(y))
                print("yHat:{0}".format(yHat))
                print("summation:{0}".format(summation))
                # 判断预测正确与否
                if yHat != y:
                    # 预测错误进行参数更新
                    w = map(lambda ww, xx:\
                                ww + self.eta * y * xx,\
                            self.wList[-1], x)
                    b = self.bList[-1] + self.eta * y
                    self.wList.append(w)
                    self.bList.append(b)
                    errNum += 1.0
                else:
                    # 预测正确追加参数占位
                    self.wList.append(self.wList[-1])
                    self.bList.append(self.bList[-1])

            # 追加保存本轮结束后的损失函数值和错分率并打印出来
            costList.append(cost)
            misRate = errNum/len(curEpochYHatList)
            misRateList.append(misRate)
            print(">>> cost:{0}".format(cost))
            print(">>> misRate:{0}".format(misRate))
            # 判断本轮训练后的所有预测是否完全正确
            if curEpochYHatList == yList:
                # 训练集全部正确则停止迭代
                break
            print

        parameterDict = dict()
        parameterDict['w'] = self.wList[-1]
        parameterDict['b'] = self.bList[-1]
        return parameterDict, costList, misRateList

    def predict(self, x, iterIdx=None):
        '''
        使用第 iterIdx (=epochIdx*len(xList)+xIdx)次更新的参数
        来预测样本 x 的类别。若未指明 iterIdx ，则使用最新一次的参
        数（即最后一次参数）来预测。
        :param x: 输入样本（一个样本）特征
        :param iterIdx: 使用第 i 次训练过程中得到的感知器参数进行
        训练
        :return: 返回预测得到的类别标签（对预测值进行符号转换）以及
        该样本的预测值
        '''
        if iterIdx is None:
            iterIdx = -1
        summation = sum(\
            map(lambda xx, ww:\
                    xx*ww,\
                x, self.wList[iterIdx])\
            )
        summation += self.bList[iterIdx]
        yHat = self.sign(summation)
        return yHat, summation

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
        plt.figure('Perceptron Cost and Mis-classification Rate',figsize=(8, 9))
        # 设定两个子图和位置关系
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        # 选择子图1并绘制损失函数值折线图及相关坐标轴
        plt.sca(ax1)
        plt.plot(xrange(1, len(costList)+1), costList, '--b*')
        plt.xlabel('Epoch No.')
        plt.ylabel('Cost')
        plt.title('Plot of Cost Function')
        plt.grid()
        ax1.legend(u"Cost", loc='best')

        # 选择子图2并绘制错分率折线图及相关坐标轴
        plt.sca(ax2)
        plt.plot(xrange(1, len(misRateList)+1), misRateList, '-r*')
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
    # 初始化参数
    dataPath = "./input"
    maxEpoch = 1000
    learningRate = 10E-2
    saveFigPath = './PerceptronPlot.png'

    # 加载训练数据
    idList, xList, yList = readDataFrom(path=dataPath)
    print("idList:{0}".format(idList))
    print("xList:{0}".format(xList))
    print("yList:{0}".format(yList))

    # 初始化感知器类
    p = Perceptron(sampleNum=len(xList),\
                   featureNum=len(xList[0]),\
                   learningRate=learningRate)

    # 打印初始时的感知器参数
    print("p.wList:{0}".format(p.wList))
    print("p.bList:{0}".format(p.bList))

    # 训练感知器
    print("====== train ======")
    parameterDict, costList, misRateList = p.train(xList=xList,\
                                                   yList=yList,\
                                                   maxEpoch=maxEpoch)

    # 打印结果
    print("====== result ======")
    print("parameterDict['w']:{0}".format(parameterDict['w']))
    print("parameterDict['b']:{0}".format(parameterDict['b']))

    print("yList:{0}".format(yList))
    print("yHatList:{0}".format(map(lambda x:\
                                        p.predict(x),\
                                    xList)))

    print("p.wList:{0}".format(p.wList))
    print("p.bList:{0}".format(p.bList))

    print("len(p.wList):{0}".format(len(p.wList)))
    print("len(p.wList):{0}".format(len(p.bList)))

    # 绘制损失函数和错分率随 epoch 变化的图像
    p.plotChart(costList=costList,\
                misRateList=misRateList,\
                saveFigPath=saveFigPath)