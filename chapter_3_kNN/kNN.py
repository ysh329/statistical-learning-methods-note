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


class kNN(object):

   def __init__(self, sampleNum, featureNum, kNum=None, distancePValue=None):
       if kNum == None:
           kNum = 1
       if distancePValue == None:
            distancePValue = 2

       self.sampleNum = sampleNum
       self.featureNum = featureNum
       self.kNum = kNum
       self.p = float(distancePValue)

   def distanceBetween(self, aList, bList, p=None):
       if p == None:
           p = self.p

       import math
       sigma = sum(\
           map(lambda aa, bb:\
                   math.pow(aa-bb, p),\
               aList, bList)\
           )
       distance = math.pow(sigma.__abs__(), 1.0/p)
       return distance

################################### PART3 TEST ########################################
# 例子
if __name__ == "__main__":
    # 参数初始化
    kNum = 2
    distancePValue = 2

    dataPath = "./input1"
    hasHeader = True
    idList, xList, yList = readDataFrom(path=dataPath,\
                                        hasHeader=hasHeader)
    print("idList:{0}".format(idList))
    print("xList:{0}".format(xList))
    print("yList:{0}".format(yList))

    #
    knn = kNN(sampleNum=len(idList),\
              featureNum=len(xList[0]),\
              kNum=kNum,\
              distancePValue=distancePValue)

    print knn.distanceBetween(aList=[1,3,5],
                        bList=[2,4,6],\
                              p=2)