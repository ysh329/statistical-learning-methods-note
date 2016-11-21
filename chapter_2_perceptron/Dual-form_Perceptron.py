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
        self.alphaList = [map(lambda i:\
                                  0.0, #random.random(),\
                              xrange(self.sampleNum))\
                          ]
        self.bList = [0.0] #[random.random()]

    def constructGramMatrix(self, xList):
        '''
        构造 Gram 矩阵
        :param xList:
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
        for epochIdx in xrange(maxEpochNum):
            print("======= epochIdx {0} =======".format(epochIdx))
            for sampleIdxI in xrange(len(xList)):
                x = xList[sampleIdxI]
                yHat, sigma = self.predict(x,\
                                           xList,\
                                           yList,\
                                           useGramMatrix=True,\
                                           sampleIdxI=sampleIdxI,\
                                           iterIdx=None)
                cost = yList[sampleIdxI] * sigma
                # 打印cost
                iterIdx = epochIdx * len(xList) + sampleIdxI
                print("== iterIdx:{0} ==".format(iterIdx))
                print("cost:{0}".format(cost))
                # 判断是否进行参数更新
                if cost <= 0:
                    nextAlpha = self.alphaList[epochIdx][sampleIdxI] + self.eta
                    nextB = self.bList[epochIdx] + self.eta * yList[sampleIdxI]
                else:
                    nextAlpha = self.alphaList[-1]
                    nextB = self.bList[-1]
                self.alphaList.append(nextAlpha)
                self.bList.append(nextB)
            print map(lambda x: self.predict(x, xList, yList), xList)


    def predict(self, x, xList, yList, useGramMatrix=False, sampleIdxI=None, iterIdx=None):
        if iterIdx is None:
            iterIdx = -1
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
                        xList[sampleIdxJ] *\
                        x,\
                    xrange(len(self.sampleNum)))\
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

################################### PART3 TEST ########################################
# 例子
if __name__ == "__main__":

    dataPath = "./input"
    learningRate = 10E-3
    maxEpochNum = 1

    idList, xList, yList = readDataFrom(path=dataPath,\
                                        hasHeader=True)
    print("idList:{0}".format(idList))
    print("xList:{0}".format(xList))
    print("yList:{0}".format(yList))

    dfp = DualFormPerceptron(sampleNum=len(xList),\
                             featureNum=len(xList[0]),\
                             learningRate=learningRate)

    dfp.constructGramMatrix(xList=xList)

    dfp.train(xList=xList,\
              yList=yList,\
              maxEpochNum=1)