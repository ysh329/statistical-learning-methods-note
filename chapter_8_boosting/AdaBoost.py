# -*- coding: utf-8 -*-
################################### PART0 DESCRIPTION #################################
# Filename: AdaBoost.py
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
        if x < self.threshold:
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
                                                        isCorrectPredict == False,\
                                                    idxAndIsCorrectPredictTupleList)
        errIdxList = map(lambda (idx, isCorrectPredict):\
                             idx,\
                         errIdxAndIsCorrectPredictTupleList)
        return yHatList, errNum, errRate, errIdxList


    def maxAndMin(self, x):
        return min(x), max(x)


class Boosting(object):
    def __init__(self, sampleNum, baseClassifierNum):
        self.sampleNum = sampleNum
        self.baseClassifierNum = baseClassifierNum

        # DataSetWeightDistribution
        self.wList = [[1.0 / self.sampleNum] * self.sampleNum] * (self.baseClassifierNum+1)
        # NormalizationFactor
        self.zList = [None] * self.baseClassifierNum
        # BaseClassifierCoefficient
        self.alphaList = [None] * self.baseClassifierNum

    def __del__(self):
        pass

    def computeBaseClassifierCoefficient(self, classifierIdx, errRate):
        self.alphaList[classifierIdx] = 1.0 / 2 * cmath.log(float(1-errRate)/errRate, cmath.e)

    def computeDataSetWeightDistribution(self, classifierIdx, yList, yHatList):
        tmpWList = map(lambda w, y, yHat:\
                           (w * cmath.exp(-self.alphaList[classifierIdx]) * y * yHat).real,\
                       self.wList[classifierIdx], yList, yHatList)
        self.computeNormalizationFactor(classifierIdx=classifierIdx, tmpWList=tmpWList)
        self.wList[classifierIdx+1] = map(lambda tmpW: tmpW / self.zList[classifierIdx], tmpWList)

    def computeNormalizationFactor(self, classifierIdx, tmpWList):
        self.zList[classifierIdx] = sum(tmpWList)

    def predict(self, x, baseClassifierList):
        baseClassifierResultList = map(lambda baseClassifier, alpha: alpha.real * baseClassifier.predict(x), baseClassifierList, self.alphaList)
        if sum(baseClassifierResultList) > 0:
            return 1
        else:
            return -1

    def train(self, xList, yList, baseClassifierList):
        yHatList = map(lambda x: self.predict(x, baseClassifierList), xList)
        isCorrectPredictList = map(lambda yHat, y: yHat == y, yHatList, yList)
        # 错分数
        errNum = isCorrectPredictList.count(False)
        # 错分率
        errRate = float(errNum) / len(xList)
        # 错分样本下标
        idxAndIsCorrectPredictTupleList = map(lambda idx, isCorrectPredict: \
                                                  (idx, isCorrectPredict), \
                                              xrange(len(xList)), isCorrectPredictList)
        print("yList:{0}".format(yList))
        print("yHatList:{0}".format(yHatList))
        print("isCorrectPredictList:{0}".format(isCorrectPredictList))
        errIdxAndIsCorrectPredictTupleList = filter(lambda (idx, isCorrectPredict): \
                                                        isCorrectPredict == False, \
                                                    idxAndIsCorrectPredictTupleList)
        errIdxList = map(lambda (idx, isCorrectPredict): \
                             idx, \
                         errIdxAndIsCorrectPredictTupleList)
        return yHatList, errNum, errRate, errIdxList


################################### PART3 TEST #######################################
# 参数初始化
dataPath = './input'
baseClassifierNum = 3
baseClassifierThresholdList = [2.5, 8.5, 5.5]

# 读取数据
idList, xList, yList = readDataFrom(dataPath, hasHeader=True)
print("id:{0}".format(idList))
print("x:{0}".format(xList))
print("y:{0}".format(yList))

# 初始化boosting类和多个分类器
boosting = Boosting(sampleNum=len(idList), baseClassifierNum=baseClassifierNum)
baseClassifierList = map(lambda threshold: BaseClassifier(xList=xList, threshold=threshold), baseClassifierThresholdList)

# 训练boosting
for classifierIdx in xrange(boosting.baseClassifierNum):
    print("classifierIdx:{0}".format(classifierIdx))
    # 分类器遍历
    baseClassifier = baseClassifierList[classifierIdx]
    yHatList, errNum, errRate, errIdxList = baseClassifier.train(xList=xList,\
                                                                 yList=yList)

    print("errNum:{0}".format(errNum))
    print("errRate:{0}".format(errRate))
    print("errIdxList:{0}".format(errIdxList))

    # 计算分类器系数
    boosting.computeBaseClassifierCoefficient(classifierIdx=classifierIdx,\
                                              errRate=errRate)

    # 计算数据集权重分布
    boosting.computeDataSetWeightDistribution(classifierIdx=classifierIdx,\
                                              yList=yList,\
                                              yHatList=yHatList)

    # 打印分类器系数
    print("boosting.alphaList[classifierIdx].real:{0}".format(boosting.alphaList[classifierIdx].real))
    print("boosting.zList[classifierIdx]:{0}".format(boosting.zList[classifierIdx]))
    print

# boosting分类器
yHatList, errNum, errRate, errIdxList = boosting.train(xList=xList,\
                                                       yList=yList, \
                                                       baseClassifierList=baseClassifierList)

print("errNum:{0}".format(errNum))
print("errRate:{0}".format(errRate))
print("errIdxList:{0}".format(errIdxList))
