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
        self.wList = [map(lambda wIdx: \
                              random.random(), \
                          xrange(featureNum))]
        self.bList = [random.random()]

    def constructGramMatrix(self, xList):
        self.gramMatrix = [range(self.sampleNum)] * self.sampleNum
        print self.gramMatrix
        for idx1 in xrange(self.sampleNum):
            for idx2 in xrange(self.sampleNum):
                # if idx1 <= idx2:
                innerProd = sum(\
                    map(lambda xx1, xx2:\
                            xx1 * xx2,\
                        xList[idx1], xList[idx2])\
                    )
                print idx1,idx2,innerProd
                self.gramMatrix[idx1][idx2] = innerProd


        for i in xrange(self.sampleNum):
            for j in xrange(self.sampleNum):
                print self.gramMatrix[i][j],
            print

        print str(self.gramMatrix)
        print xList[0], xList[1]
        print self.gramMatrix[0][1]


if __name__ == "__main__":
    dataPath = "./input"
    learningRate = 10E-3

    idList, xList, yList = readDataFrom(path=dataPath,\
                                        hasHeader=True)
    print("idList:{0}".format(idList))
    print("xList:{0}".format(xList))
    print("yList:{0}".format(yList))

    dfp = DualFormPerceptron(sampleNum=len(xList),\
                             featureNum=len(xList[0]),\
                             learningRate=learningRate)

    dfp.constructGramMatrix(xList=xList)