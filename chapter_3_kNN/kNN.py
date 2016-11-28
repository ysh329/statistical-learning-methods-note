# -*- coding: utf-8 -*-
################################### PART0 DESCRIPTION #################################
# Filename: kNN.py
# Description:
#

# E-mail: ysh329@sina.com
# Create: 2016-11-24 23:20:10
# Last:
__author__ = 'yuens'


################################### PART1 IMPORT ######################################
import math
import matplotlib.pylab as plt

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
            rawDataWithoutHeader = rawData[1:]
        else:
            print("header:None")
            rawDataWithoutHeader = rawData
        cleanData = map(lambda recordList: \
                            map(int, recordList),\
                        rawDataWithoutHeader)
        idList = map(lambda r: r[0], cleanData)
        xList = map(lambda r: tuple(r[1:len(r)-1]), cleanData)
        yList = map(lambda r: r[-1], cleanData)
        return idList, xList, yList


class kNN(object):
    '''
    k最近邻模型的类。无监督模型。
    '''
    def __init__(self, sampleNum, featureNum, k=None, distancePValue=None):
        '''
        初始化模型参数
        :param sampleNum: 训练集样本个数
        :param featureNum: 每个样本的特征个数
        :param k: 分类基于最近的 k 个样本
        :param distancePValue: $L_p$ 距离参数
        '''
        # 参数检查
        if k == None:
            k = 1
        if distancePValue == None:
            distancePValue = 2

        self.sampleNum = sampleNum
        self.featureNum = featureNum
        self.k = k
        self.p = float(distancePValue)
        self.distMatrix = dict()

    def constructDistMatrix(self, xList, p=None):
        '''
        构建距离矩阵，类似临街矩阵，但不是，其索引是通过两个实例点
        作为索引进行的，而不是下标序号。构建过程中借助计算两个点距
        离的函数。
        :param xList: 训练样本特征
        :param p: $L_p$ 距离参数
        :return:
        '''
        if p == None:
            p = self.p
        # 初始化
        for x1Idx in xrange(len(xList)):
            x1 = xList[x1Idx]
            if not self.distMatrix.has_key(x1):
                self.distMatrix[x1] = dict()
            for x2Idx in xrange(len(xList)):
                x2 = xList[x2Idx]
                if not self.distMatrix[x1].has_key(x2):
                    self.distMatrix[x1][x2] = 0.0
       # 计算距离
        for x1Idx in xrange(len(xList)):
            for x2Idx in xrange(len(xList)):
                x1 = xList[x1Idx]
                x2 = xList[x2Idx]
                if x1Idx != x2Idx and self.distMatrix[x1][x2] == 0.0:
                    self.distMatrix[x1][x2] = self.distanceBetween(aList=x1,\
                                                                   bList=x2,\
                                                                   p=p)
                    self.distMatrix[x2][x1] = self.distMatrix[x1][x2]

    def distanceBetween(self, aList, bList, p=None):
        '''
        计算两个点，表示为 aList 与 bList，二者之间的 $L_p$ 距离。
        :param aList: 第一个点的特征
        :param bList: 第二个点的特征
        :param p: $L_p$ 距离参数
        :return: 返回两个点之间的距离
        '''
        if p == None:
             p = self.p
        sigma = sum(\
             map(lambda aa, bb:\
                     math.pow(aa-bb, p),\
                 aList, bList)\
             )
        distance = math.pow(sigma.__abs__(), 1.0/p)
        return distance

    def chooseK(self, xList, yList, p=None):
        '''
        基于训练数据，选择最合适的 k 值。会对 k 从 1 到所有样本数进行遍历，
         统计不同 k 值时分类正确的点数，最终的 k 选择分类错误最少的。
        :param xList: 训练样本特征
        :param yList: 训练样本标签
        :param p: $L_p$ 距离参数
        :return: 返回分类错误最少的 k 值
        '''
        # 参数检查默认为2
        if p == None:
            p = self.p
        kList = range(1, len(xList))
        misClassDict = dict()
        # 遍历k
        for kIdx in xrange(len(kList)):
            k = kList[kIdx]
            misClassDict[k] = 0
            # 选择当前k下，每个样本的yHat
            for xIdx in xrange(len(xList)):
                x = xList[xIdx]
                xAndDistAndYTupList = map(lambda (x2, dist):\
                                              (x2, dist, yList[xList.index(x2)]),\
                                          self.distMatrix[x].iteritems())
                xAndDistAndYTupList.sort(key=lambda (x2, dist, y): dist,\
                                         reverse=False)
                xAndDistAndYTupList = filter(lambda (x2, dist, y): x2 != x, xAndDistAndYTupList)
                # 统计当前样本的k近邻的类别
                yHatDict = dict()
                yHatList = map(lambda (x2, dist, y):\
                                   y,\
                               xAndDistAndYTupList[:k])
                for idx in xrange(len(yHatList)):
                    yHat = yHatList[idx]
                    if yHatDict.has_key(yHat):
                        yHatDict[yHat] += 1
                    else:
                        yHatDict[yHat] = 1
                yHatAndCountList = map(lambda (yHat, count):\
                                           (yHat, count),\
                                       yHatDict.iteritems())
                yHatAndCountList.sort(key=lambda (yHat, count): count,\
                                      reverse=True)
                xsYHat = yHatAndCountList[0][0]
                if yList[xIdx] != xsYHat:
                    misClassDict[k] += 1
        # 选择错误最少的k
        kAndMisNumList = map(lambda (k, misNum):\
                                 (k, misNum),\
                             misClassDict.iteritems())
        kAndMisNumList.sort(key=lambda (k, misNum): misNum,\
                            reverse=False)
        bestK = kAndMisNumList[0][0]
        return bestK, misClassDict

    def predict(self, x, xList, yList, p=None):
        '''
        预测新输入样本 x 的类别，其中 xList 与 yList 分别是训练样本的
        特征和类别标签， p 为 $L_P$ 距离的参数（默认为None，会被设定为
        2）。
        :param x: 新输入样本的特征
        :param xList: 训练样本特征
        :param yList: 训练样本标签
        :param p: $L_p$ 距离参数
        :return: 返回 x 预测出的类别
        '''
        # 检查参数
        if p == None:
            p = self.p
        # 整合距离矩阵生成所需要的数据格式
        xAndXXAndDistAndYTupList = map(lambda xx, y:\
                                           (x,\
                                            xx,\
                                            self.distanceBetween(aList=x,\
                                                                 bList=xx,\
                                                                 p=p),
                                            y),\
                                       xList, yList)
        xAndXXAndDistAndYTupList.sort(key=lambda (x, xx, dist, y): dist,\
                                      reverse=False)
        # 根据其他样本点距离 x 的远近程度统计类别数目
        yDict = {}
        for idx in xrange(self.k):
            # (x, xx, dist, y)
            y = xAndXXAndDistAndYTupList[idx][3]
            if yDict.has_key(y):
                yDict[y] += 1
            else:
                yDict[y] = 1
        yAndCountTupList = map(lambda (y, count):\
                                   (y, count),\
                               yDict.iteritems())
        yAndCountTupList.sort(key=lambda (y, count): count,\
                              reverse=True)
        yHat = yAndCountTupList[0][0]
        return yHat

    def plotScatter(self, xList, yList, saveFigPath):
        '''
        根据特征数据 xList 及其类别 yList 绘制散点图，并将绘制出的
        散点图保存在 saveFigPath 路径下。
        :param xList: 样本特征
        :param yList: 样本类别
        :param saveFigPath: 保存散点图的路径
        :return:
        '''
        # 判断特征是否大于等于二维
        # 如果样本的特征大于等于 2
        # 那么仅可视化前面 2 维度的数据
        if len(xList[0]) >= 2:
            x1List = map(lambda x: x[0], xList)
            x2List = map(lambda x: x[1], xList)
        else:
            # 1 或 2 维数据都可视化为 2 维
            x1List = x2List = map(lambda x: x[0], xList)
        # 新建画布
        scatterFig= plt.figure(saveFigPath)
        # 预定义：颜色初始化
        colorDict = {-1: 'm', 1: 'r', 2: 'b', 3: 'pink', 4: 'orange'}
        # 绘制每个点
        map(lambda idx: \
                plt.scatter(x1List[idx], \
                            x2List[idx], \
                            marker='o', \
                            color=colorDict[yList[idx]], \
                            label=yList[idx]), \
            xrange(len(x1List)))
        # 给每种类别加上标注
        # ySet = set(yList)
        # map(lambda y: \
        #         plt.legend(str(y), \
        #                    loc='best'), \
        #     ySet)

        # 设定其他属性并保存图像后显示
        plt.title(saveFigPath)
        plt.xlabel(r'$x^1$')
        plt.ylabel(r'$x^2$')
        plt.grid(True)
        plt.savefig(saveFigPath)
        plt.show()

    def plotKChart(self, misClassDict, saveFigPath):
        kList = []
        misRateList = []
        for k, misClassNum in misClassDict.iteritems():
            kList.append(k)
            misRateList.append(1.0 - 1.0/k*misClassNum)

        fig = plt.figure(saveFigPath)
        plt.plot(kList, misRateList, 'r--')
        plt.title(saveFigPath)
        plt.xlabel('k Num.')
        plt.ylabel('Misclassified Rate')
        plt.legend(saveFigPath)
        plt.grid(True)
        plt.savefig(saveFigPath)
        plt.show()

################################### PART3 TEST ########################################
# 例子
if __name__ == "__main__":
    # 参数初始化
    k = 2
    distancePValue = 2
    dataPath = "./input1"
    hasHeader = True
    saveScatterFigPath = u"k-Nearest Neighbor Scatter Plot"
    saveKChartFigPath = u"k-Nearest Neighbor's K Chart"

    # 读取数据
    idList, xList, yList = readDataFrom(path=dataPath,\
                                        hasHeader=hasHeader)
    print("idList:{0}".format(idList))
    print("xList:{0}".format(xList))
    print("yList:{0}".format(yList))

    # 实例化最近邻类
    knn = kNN(sampleNum=len(idList),\
              featureNum=len(xList[0]),\
              k=k,\
              distancePValue=distancePValue)

    # 初始化距离矩阵并完成两点间距离计算
    knn.constructDistMatrix(xList=xList)

    # 预测一个新样本点的所属类别
    newX = (1, 3)
    newYHat = knn.predict(x=newX,\
                          xList=xList,\
                          yList=yList)
    print("newYHat:{0}".format(newYHat))

    # 基于数据选择一个最合适的 k 值
    bestK, misClassDict = knn.chooseK(xList=xList,\
                                      yList=yList)
    print("bestK:{0}".format(bestK))
    knn.plotKChart(misClassDict=misClassDict,\
                   saveFigPath=saveKChartFigPath)

    # 绘制散点图
    knn.plotScatter(xList=xList,\
                    yList=yList,\
                    saveFigPath=saveScatterFigPath)