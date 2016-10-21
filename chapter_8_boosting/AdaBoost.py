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
        初始化分类器，设置阈值。
        :param threshold: float 分类阈值
        '''
        self.POSITIVE_LABEL = 1
        self.NEGATIVE_LABEL = -1

        self.threshold = threshold

    def predict(self, x, reverse=False):
        '''
        根据输入的一个样本特征值进行类别预测。
        :param x: 输入的一个样本的特征
        :param reverse: 类别对于阈值的方向（大于小于）
        :return:
        '''
        if reverse:
            return self.POSITIVE_LABEL if x > self.threshold else self.NEGATIVE_LABEL
        else:
            return self.POSITIVE_LABEL if x < self.threshold else self.NEGATIVE_LABEL

    def train(self, xList, yList, reverse=False):
        '''
        对多个样本的类别进行预测，并返回预测值(yHatList)，错分样本个数(errNum)， 错分率(errRate)， 错分样本下标(errIdxList)。
        :param xList: 多个样本的特征的列表
        :param yList: 多个样本的实际标签
        :param reverse: 类别对于阈值的方向（大于小于）
        :return:
        '''
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
    '''
    多模型集成。
    '''
    def __init__(self, sampleNum, baseClassifierNum):
        '''
        类boosting实例化，关键参数初始化。
        :param sampleNum: 训练样本数目，后面需要初始化每个样本的权重
        :param baseClassifierNum: 基分类器数目
        '''
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
        '''
        计算分类器错误率（不是常见的错误率），该错误率参考了每个样本的权重。
        :param classifierIdx: 当前正在训练的基分类器下标(从0开始)
        :param yHatList: 当前基分类器对于xList的预测值yHatList
        :param yList: 所有训练样本的真实类别yList
        :return:
        '''
        self.eList[classifierIdx] = sum(\
            map(lambda xIdx, yHat, y:\
                    self.wList[classifierIdx][xIdx] if yHat != y else 0,\
                xrange(len(yHatList)), yHatList, yList)\
            )

    def computeBaseClassifierCoefficient(self, classifierIdx):
        '''
        输入当前正在训练的基分类器下标(从0开始)，计算当前的基分类器系数 alpha 。
        :param classifierIdx: 当前分类器下标(从0开始)
        :return:
        '''
        self.alphaList[classifierIdx] = (1.0 / 2 * \
                                         cmath.log((1.0-self.eList[classifierIdx])/self.eList[classifierIdx], cmath.e)\
                                         ).real

    def computeDataSetWeightDistribution(self, classifierIdx, yList, yHatList):
        '''
        输入当前正在训练的分类器下标，样本真实值和预测值，计算训练数据集样本的权重分布。
        :param classifierIdx: 当前正在训练的分类器下标(从0开始)
        :param yList: 训练样本真实类别
        :param yHatList: 训练样本基于当前分类器的预测值
        :return:
        '''
        self.wList[classifierIdx+1] = map(lambda w, y, yHat:\
                                              (w / self.zList[classifierIdx] *\
                                               cmath.exp(-self.alphaList[classifierIdx] * y * yHat)).real,\
                                          self.wList[classifierIdx], yList, yHatList)

    def computeNormalizationFactor(self, classifierIdx, yList, yHatList):
        '''
        根据当前正在使用的分类器(下标，从0开始)、真实类别、预测类别，计算当前分类器在训练数据上的规范化因子。
        :param classifierIdx: 当前正在训练的分类器下标(从0开始)
        :param yList: 训练样本真实类别
        :param yHatList: 训练样本基于当前分类器的预测值
        :return:
        '''
        tmpZList = map(lambda w, y, yHat:\
                           w * cmath.exp(-self.alphaList[classifierIdx] * y * yHat),\
                       self.wList[classifierIdx], yList, yHatList)
        self.zList[classifierIdx] = sum(tmpZList).real

    def predict(self, x, baseClassifierList, baseClassifierNum, baseClassifierReverseList=None,\
                boostingClassifierReverse=False):
        '''
        基于baseClassifierList中的前baseClassifierNum个分类器，来预测样本x的类别。
        :param x: 待预测的样本特征
        :param baseClassifierList: 多个分类器对象的列表
        :param baseClassifierNum: 当前用到的分类器数目，从第一个开始算
        :param baseClassifierReverseList: 类别对于阈值的方向（大于小于），多个基分类器所以是一个列表
        :param boostingClassifierReverse: 类别对于阈值的方向（大于小于），集成的分类器所以是一个
        :return: 当前样本x所属的类别
        '''
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
        '''
        使用predict函数基于多个基分类器，批量预测样本的类别。
        :param xList: 多个样本的特征的列表
        :param yList: 多个样本的实际标签
        :param baseClassifierList: 多个分类器对象的列表
        :param baseClassifierNum: 当前用到的分类器数目，从第一个开始算
        :param baseClassifierReverseList: 类别对于阈值的方向（大于小于），多个基分类器所以是一个列表
        :param boostingClassifierReverse: 类别对于阈值的方向（大于小于），集成的分类器所以是一个
        :return: 批量样本的预测结果yHatList、错分数errNum、错误率errRate、错分样本的下标errIdxList
        '''
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
        '''
        训练集成模型。
        :param xList: 多个样本的特征的列表
        :param yList: 多个样本的实际标签
        :param baseClassifierReverseList: 类别对于阈值的方向（大于小于），多个基分类器所以是一个列表
        :param boostingClassifierReverse: 类别对于阈值的方向（大于小于），集成的分类器所以是一个
        :return: dict，返回最终训练得到的参数。
        '''
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
        parameterDict['alphaList'] = self.alphaList
        parameterDict['z'] = self.zList[classifierIdx]
        parameterDict['e'] = self.eList[classifierIdx]
        return parameterDict


################################### PART3 TEST #######################################
# 例子
if __name__ == "__main__":
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


    # boosting参数结果
    print("----- final boosting parameters -----")
    print("parameterDict['w']:{0}".format(parameterDict['w']))
    print("parameterDict['e']:{0}".format(parameterDict['e']))
    print("parameterDict['alphaList']:{0}".format(parameterDict['alphaList']))
    print("parameterDict['z']:{0}".format(parameterDict['z']))
