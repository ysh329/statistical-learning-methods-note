# -*- coding: utf-8 -*-
################################### PART0 DESCRIPTION #################################
# Filename: adaboost.py
# Description:
#

# E-mail: ysh329@sina.com
# Create: 2016-10-17 15:06:11
# Last:
__author__ = 'yuens'


################################### PART1 IMPORT ######################################
import random
import cmath

################################### PART2 CLASS && FUNCTION ###########################

def readDataFrom(path, hasHeader=True):
    with open(path, 'r') as f:
        rawData = map(lambda line: line.replace("\n", "").split("   "), f.readlines())
        if hasHeader:
            header = rawData[0]
            print("header:{0}".format(header))
            cleanData = map(lambda recordList:\
                           (int(recordList[0]), int(recordList[1]), int(recordList[2])),\
                       rawData[1:])
        else:
            print("Header:None")
            cleanData = map(lambda recordList:\
                           (int(recordList[0]), int(recordList[1]), int(recordList[2])),\
                       rawData)
        idList = map(lambda r: r[0], cleanData)
        xList = map(lambda r: r[1], cleanData)
        yList = map(lambda r: r[2], cleanData)
        if len(idList) == len(xList) == len(yList):
            return(idList, xList, yList)
        else:
            print("Error: 样本特征有缺失")
    return idList, xList, yList

class BaseClassifier(object):
    def __init__(self, xList=None, threshold=None):
        self.positiveLabel = 1
        self.negativeLabel = -1
        if threshold == None:
            begin, end = self.maxAndMin(xList)
            self.threshold = random.randint(begin, end)
        else:
            self.threshold = threshold

    def __del__(self):
        pass

    def setThreshold(self, threshold):
        self.threshold = threshold

    def updateThreshold(self, delta):
        newThreshold = self.threshold + delta
        self.setThreshold(newThreshold)

    def getThreshold(self):
        return self.threshold

    def predict(self, x):
        if x > self.threshold:
            return self.positiveLabel
        else:
            return self.negativeLabel

    def train(self, xList, yList):
        yHatList = map(lambda x: self.predict(x), xList)
        isCorrectPredictList = map(lambda yHat, y: yHat == y, yHatList, yList)
        # 错分数
        errNum = isCorrectPredictList.count(False)
        # 错分率
        errRate = float(errNum) / len(xList)
        # 错分样本下标
        idxAndIsCorrectPredictTupleList = map(lambda idx, isCorrectPredict:\
                                                  (idx, isCorrectPredict),\
                                              xrange(len(xList)), isCorrectPredictList)
        errIdxAndIsCorrectPredictTupleList = filter(lambda (idx, isCorrectPredict):\
                                                        isCorrectPredict == True,\
                                                    idxAndIsCorrectPredictTupleList)
        errIdxList = map(lambda (idx, isCorrectPredict):\
                             idx,\
                         errIdxAndIsCorrectPredictTupleList)
        return errNum, errRate, errIdxList


    def maxAndMin(self, x):
        return min(x), max(x)


class Boosting(object):
    def __init__(self, sampleNum, baseClassifierNum):
        self.sampleNum = sampleNum
        self.baseClassifierNum = baseClassifierNum

        self.wList = [[float(1) / self.sampleNum] * self.sampleNum] * self.baseClassifierNum
        self.zList = [None] * self.baseClassifierNum
        self.alphaList = [None] * self.baseClassifierNum

    def __del__(self):
        pass

    def computeBaseClassifierCoefficient(self, classifierIdx, errRate):
        self.alphaList[classifierIdx] = float(1/2) * cmath.log(float(1-errRate)/errRate, cmath.e)

    def computeDataSetWeightDistribution(self):
        pass

    def computeNormalizationFactor(self):
        pass





################################### PART3 TEST #######################################
# 参数初始化
dataPath = './input'
baseClassifierNum = 3

# 读取数据
idList, xList, yList = readDataFrom(dataPath, hasHeader=True)
print("id:{0}".format(idList))
print("x:{0}".format(xList))
print("y:{0}".format(yList))

# 分类器1
baseClassifier1 = BaseClassifier(xList=xList)
errNum, errRate, errIdxList = baseClassifier1.train(xList=xList, yList=yList)
print("errNum:{0}".format(errNum))
print("errRate:{0}".format(errRate))
print("errIdxList:{0}".format(errIdxList))


boosting = Boosting(sampleNum=len(idList), baseClassifierNum=baseClassifierNum)


