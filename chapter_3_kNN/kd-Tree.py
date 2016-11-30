# -*- coding: utf-8 -*-
################################### PART0 DESCRIPTION #################################
# Filename: kd-Tree.py
# Description:
#

# E-mail: ysh329@sina.com
# Create: 2016-11-29 09:58:21
# Last:
__author__ = 'yuens'


################################### PART1 IMPORT ######################################


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

class Node(object):
    '''
    构造结点并生成二叉树。
    '''
    def __init__(self, xList):
        '''
        基于列表数据 xList 构造二叉树，
        :param xList:
        '''
        if len(xList)==0:
            return None
        self.value = median(xList)
        self.left = None if len(xList)<=1 else Node(xList=filter(lambda v:\
                                                                     v <= self.value,\
                                                                 xList)\
                                                    )
        self.right = None if len(xList)<=1 else Node(xList=filter(lambda v:\
                                                                      v > self.value,\
                                                                  xList)\
                                                     )

    def midTravel(self):
        self.midTravel()
        print("self.value:{0}".format(self.value))

def median(xList):
    '''
    计算列表 x的中位数并返回。若列表中元素个数为偶数，则中位数为中间两个数的均值，
    如果列表中元素个数为奇数，则中位数为列表中间的元素。
    :param l: 输入列表 l
    :return: 返回列表 l 的中位数
    '''
    if not isinstance(xList, list):
        print("input value is not list variable.")
        return -1
    xList.sort()
    if len(xList) % 2 == 0:
        return (xList[len(xList) / 2] + xList[(len(xList) - 1) / 2]) / 2.0
    else:
        return xList[(len(xList) - 1) / 2]

################################### PART3 TEST ########################################
# 例子
if __name__ == "__main__":
    # 参数初始化
    dataPath = "./input1"
    hasHeader = True

    # 读取数据
    idList, xList, yList = readDataFrom(path=dataPath,\
                                        hasHeader=hasHeader)
    print("idList:{0}".format(idList))
    print("xList:{0}".format(xList))
    print("yList:{0}".format(yList))

    # 实例化kd-Tree
    root = Node(xList)

    print median(l=[1,2,3])
    print median(l=[1,2,3,4])
    xx = [1,2]
    print median(xx)
    print filter(lambda x: x>=median(xx), xx)
    print filter(lambda x: x<median(xx), xx)


    xList = [1,2,3,4,5,6,7,8,9,]
    N = Node(xList)
    print N.value
    print
    print N.left