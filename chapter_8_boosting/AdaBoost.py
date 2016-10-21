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
import cmath

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
    '''
    二类线性分类器，根据阈值对输入数据进行类别判断。
    '''
    def __init__(self, threshold=None):
        '''

        :param xList:
        :param yList:
        :param threshold:
        '''
        self.POSITIVE_LABEL = 1
        self.NEGATIVE_LABEL = -1

        self.threshold = threshold

    def predict(self, x, reverse=False):
        if reverse:
            return self.POSITIVE_LABEL if x > self.threshold else self.NEGATIVE_LABEL
        else:
            return self.POSITIVE_LABEL if x < self.threshold else self.NEGATIVE_LABEL

    def train(self, xList, yList, reverse=False):
        yHatList = map(lambda x: self.predict(x, reverse=reverse), xList)
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

class Boosting(object):
    def __init__(self, sampleNum, baseClassifierNum):
        self.sampleNum = sampleNum
        self.baseClassifierNum = baseClassifierNum

        self.POSITIVE_LABEL = 1
        self.NEGATIVE_LABEL = -1

        # Error Rate
        self.eList = [None] * (self.baseClassifierNum)
        # DataSetWeightDistribution
        self.wList = [[1.0 / self.sampleNum] * self.sampleNum] * (self.baseClassifierNum+1)
        # NormalizationFactor
        self.zList = [None] * self.baseClassifierNum
        # BaseClassifierCoefficient
        self.alphaList = [None] * self.baseClassifierNum

    def computeClassifierErrorRate(self, classifierIdx, yHatList, yList):
        self.eList[classifierIdx] = sum(\
            map(lambda xIdx, yHat, y:\
                    self.wList[classifierIdx][xIdx] if yHat != y else 0,\
                xrange(len(yHatList)), yHatList, yList)\
            )

    def computeBaseClassifierCoefficient(self, classifierIdx):
        self.alphaList[classifierIdx] = (1.0 / 2 * \
                                         cmath.log((1.0-self.eList[classifierIdx])/self.eList[classifierIdx], cmath.e)\
                                         ).real

    def computeDataSetWeightDistribution(self, classifierIdx, yList, yHatList):
        self.wList[classifierIdx+1] = map(lambda w, y, yHat:\
                                              (w / self.zList[classifierIdx] *\
                                               cmath.exp(-self.alphaList[classifierIdx] * y * yHat)).real,\
                                          self.wList[classifierIdx], yList, yHatList)

    def computeNormalizationFactor(self, classifierIdx, yList, yHatList):
        tmpZList = map(lambda w, y, yHat:\
                           w * cmath.exp(-self.alphaList[classifierIdx] * y * yHat),\
                       self.wList[classifierIdx], yList, yHatList)
        self.zList[classifierIdx] = sum(tmpZList).real

    def predict(self, x, baseClassifierList, baseClassifierNum, baseClassifierReverseList=None,\
                boostingClassifierReverse=False):
        if baseClassifierReverseList is None:
            baseClassifierReverseList = [False] * len(baseClassifierList)
        baseClassifierResultList = map(lambda baseClassifier, alpha, baseClassifierReverse:\
                                           alpha * baseClassifier.predict(x, baseClassifierReverse),\
                                       baseClassifierList[:baseClassifierNum], self.alphaList[:baseClassifierNum],\
                                       baseClassifierReverseList[:baseClassifierNum])
        if boostingClassifierReverse:
            return self.POSITIVE_LABEL if sum(baseClassifierResultList) > 0 else self.NEGATIVE_LABEL
        else:
            return self.POSITIVE_LABEL if sum(baseClassifierResultList) < 0 else self.NEGATIVE_LABEL

    def train(self, xList, yList, baseClassifierList, baseClassifierNum, baseClassifierReverseList=None,\
              boostingClassifierReverse=False):
        yHatList = map(lambda x:\
                           self.predict(x,\
                                        baseClassifierList,\
                                        baseClassifierNum,\
                                        baseClassifierReverseList,\
                                        boostingClassifierReverse\
                                        ),\
                       xList)
        isCorrectPredictList = map(lambda yHat, y: yHat == y, yHatList, yList)
        # 错分数
        errNum = isCorrectPredictList.count(False)
        # 错分率
        errRate = float(errNum) / len(xList)
        # 错分样本下标
        idxAndIsCorrectPredictTupleList = map(lambda idx, isCorrectPredict: \
                                                  (idx, isCorrectPredict), \
                                              xrange(len(xList)), isCorrectPredictList)
        errIdxAndIsCorrectPredictTupleList = filter(lambda (idx, isCorrectPredict): \
                                                        isCorrectPredict == False, \
                                                    idxAndIsCorrectPredictTupleList)
        errIdxList = map(lambda (idx, isCorrectPredict): \
                             idx, \
                         errIdxAndIsCorrectPredictTupleList)
        return yHatList, errNum, errRate, errIdxList

    def boostTraining(self, xList, yList, baseClassifierReverseList, boostingClassifierReverse):
        for classifierIdx in xrange(self.baseClassifierNum):
            # 初始化参数
            baseClassifierReverse = baseClassifierReverseList[classifierIdx]
            usedBaseClassifierNum = classifierIdx + 1

            print("----- baseClassifier {0} -----".format(classifierIdx))
            # 依次遍历分类器
            baseClassifier = baseClassifierList[classifierIdx]
            yHatList, errNum, errRate, errIdxList = baseClassifier.train(xList=xList, \
                                                                         yList=yList, \
                                                                         reverse=baseClassifierReverse)

            # 计算分类器误差
            print("errNum:{0}".format(errNum))
            print("errRate:{0}".format(errRate))
            print("errIdxList:{0}".format(errIdxList))

            # 计算分类器误差率
            self.computeClassifierErrorRate(classifierIdx=classifierIdx, \
                                            yHatList=yHatList, \
                                            yList=yList)

            # 计算分类器系数
            self.computeBaseClassifierCoefficient(classifierIdx=classifierIdx)

            # 计算规范化因子
            self.computeNormalizationFactor(classifierIdx=classifierIdx, \
                                            yList=yList, \
                                            yHatList=yHatList)

            # 计算数据集权重分布
            self.computeDataSetWeightDistribution(classifierIdx=classifierIdx, \
                                                  yList=yList, \
                                                  yHatList=yHatList)

            # 打印分类器系数
            print("----- boosting -----")
            print("boosting.wList[classifierIdx]:{0}".format(boosting.wList[classifierIdx]))
            print("boosting.alphaList[classifierIdx]:{0}".format(boosting.alphaList[classifierIdx]))
            print("boosting.zList[classifierIdx]:{0}".format(boosting.zList[classifierIdx]))
            print("boosting.eList[classifierIdx]:{0}".format(boosting.eList[classifierIdx]))

            yHatList, errNum, errRate, errIdxList = self.train(xList=xList, \
                                                               yList=yList, \
                                                               baseClassifierList=baseClassifierList, \
                                                               baseClassifierNum=usedBaseClassifierNum, \
                                                               baseClassifierReverseList=baseClassifierReverseList[\
                                                                                         :usedBaseClassifierNum], \
                                                               boostingClassifierReverse=boostingClassifierReverse)
            print("errIdxList:{0}".format(errIdxList))
            print("yHatList:{0}".format(yHatList))
            print("errRate:{0}".format(errRate))
            print

        # 对最终的参数打包成一个字典
        parameterDict = dict()
        parameterDict['w'] = self.wList[classifierIdx + 1]
        parameterDict['alpha'] = self.alphaList[classifierIdx]
        parameterDict['z'] = self.zList[classifierIdx]
        parameterDict['e'] = self.eList[classifierIdx]
        return parameterDict


################################### PART3 TEST #######################################
# 参数初始化
dataPath = './input'
baseClassifierNum = 3
baseClassifierThresholdList = [2.5, 8.5, 5.5]
baseClassifierReverseList = [False, False, True]
boostingClassifierReverse = True

# 读取数据
idList, xList, yList = readDataFrom(dataPath, hasHeader=True)
print("id:{0}".format(idList))
print("x:{0}".format(xList))
print("y:{0}".format(yList))

# 初始化boosting类和多个分类器
boosting = Boosting(sampleNum=len(idList), baseClassifierNum=baseClassifierNum)
baseClassifierList = map(lambda threshold:\
                             BaseClassifier(threshold=threshold),\
                         baseClassifierThresholdList)

parameterDict = boosting.boostTraining(xList=xList,\
                                       yList=yList,\
                                       baseClassifierReverseList=baseClassifierReverseList,\
                                       boostingClassifierReverse=boostingClassifierReverse)

# # 训练boosting
# for classifierIdx in xrange(boosting.baseClassifierNum):
#     # 初始化参数
#     baseClassifierReverse = baseClassifierReverseList[classifierIdx]
#     usedBaseClassifierNum = classifierIdx + 1
#
#     print("----- baseClassifier {0} -----".format(classifierIdx))
#     # 依次遍历分类器
#     baseClassifier = baseClassifierList[classifierIdx]
#     yHatList, errNum, errRate, errIdxList = baseClassifier.train(xList=xList,\
#                                                                  yList=yList,\
#                                                                  reverse=baseClassifierReverse)
#
#     # 计算分类器误差
#     print("errNum:{0}".format(errNum))
#     print("errRate:{0}".format(errRate))
#     print("errIdxList:{0}".format(errIdxList))
#
#     # 计算分类器误差率
#     boosting.computeClassifierErrorRate(classifierIdx=classifierIdx,\
#                                         yHatList=yHatList,\
#                                         yList=yList)
#
#     # 计算分类器系数
#     boosting.computeBaseClassifierCoefficient(classifierIdx=classifierIdx)
#
#     # 计算规范化因子
#     boosting.computeNormalizationFactor(classifierIdx=classifierIdx,\
#                                         yList=yList,\
#                                         yHatList=yHatList)
#
#     # 计算数据集权重分布
#     boosting.computeDataSetWeightDistribution(classifierIdx=classifierIdx,\
#                                               yList=yList,\
#                                               yHatList=yHatList)
#
#     # 打印分类器系数
#     print("----- boosting -----")
#     print("boosting.wList[classifierIdx]:{0}".format(boosting.wList[classifierIdx]))
#     print("boosting.alphaList[classifierIdx]:{0}".format(boosting.alphaList[classifierIdx]))
#     print("boosting.zList[classifierIdx]:{0}".format(boosting.zList[classifierIdx]))
#     print("boosting.eList[classifierIdx]:{0}".format(boosting.eList[classifierIdx]))
#
#     yHatList, errNum, errRate, errIdxList = boosting.train(xList=xList,\
#                                                            yList=yList,\
#                                                            baseClassifierList=baseClassifierList,\
#                                                            baseClassifierNum=usedBaseClassifierNum,\
#                                                            baseClassifierReverseList=baseClassifierReverseList[:usedBaseClassifierNum],\
#                                                            boostingClassifierReverse=boostingClassifierReverse)
#     print("errIdxList:{0}".format(errIdxList))
#     print("yHatList:{0}".format(yHatList))
#     print("errRate:{0}".format(errRate))
#     print
#
# # boosting分类器
# yHatList, errNum, errRate, errIdxList = boosting.train(xList=xList,\
#                                                        yList=yList, \
#                                                        baseClassifierList=baseClassifierList,\
#                                                        baseClassifierNum=boosting.baseClassifierNum,\
#                                                        baseClassifierReverseList=baseClassifierReverseList,\
#                                                        boostingClassifierReverse=boostingClassifierReverse)
#
# # boosting结果
# print("----- final boosting -----")
# print("yHatList:{0}".format(yHatList))
# print("errNum:{0}".format(errNum))
# print("errRate:{0}".format(errRate))
# print("errIdxList:{0}".format(errIdxList))
# print("boosting.alphaList:{0}".format(boosting.alphaList))
