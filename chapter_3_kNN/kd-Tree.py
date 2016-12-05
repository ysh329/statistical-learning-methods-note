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


class kdTree(object):
    '''
    构造结点并生成 kd 树，kd 树是一种二叉树，构造方法与一般的二叉树差别不大，
    每个结点根据当前样本集中的当前特征中位数作为划分为两部分，两部分即左右子
    结点。递归生成子结点，直到样本集中没有实例，递归完也就构造完了 kd 树。
    '''
    class kdNode(object):
        '''
        kd 树的结点
        '''
        def __init__(self, value=None, left=None, right=None, parent=None, featureIdx=None, layerIdx=None):
            self.value = value
            self.featureIdx = featureIdx
            self.layerIdx = layerIdx

            self.left = left
            self.right = right
            self.parent = parent

    def __init__(self):
        pass

    def createKDTree(self, xList, layerIdx=0, featureIdx=0):
        '''
        创建 kd 树，基于传入的数据集构建。
        :param xList: 输入数据集，包含所有样本的特征
        :param layerIdx: 起始层下标，默认从 0 开始
        :param featureIdx: 起始层维度划分的特征下标，默认从 0 开始
        :return seed: 返回根结点
        '''
        seed = kdTree.kdNode()
        self.treeGrowth(root=seed, xList=xList, layerIdx=layerIdx, featureIdx=featureIdx)
        return seed

    def treeGrowth(self, root, xList, layerIdx, featureIdx):
        '''
        基于列表样本集数据 xList 构造 kd 二叉树，根据当前样本集中当前特征构造。
        注意：每个实例样本都是一个 list 类型，换句话说 xList 是一个二维 list 。
        :param xList: 包含所有样本的二维 list
        :param layerIdx: 要构造的层数（由 1 开始）
        :param featureIdx: 要构造的结点的 value 选取的参照是第 featureIdx 个特征
                            featureIdx 从 0 开始
        '''
        # 当前分支样本遍历完则结束否则新建结点继续构建
        if len(xList)==0:
            return None

        # 按照特征下标 featureIdx 进行排序找中位数下标
        if featureIdx == len(xList[0]):
            featureIdx = 0
        root.featureIdx = featureIdx
        featureIdx += 1

        # 对样本按照当前维度所对应的特征进行升序排序
        # 找到当前划分维度的特征的中位数样本下标
        # 并非严格意义的中位数
        xList.sort(key=lambda x: x[root.featureIdx],\
                   reverse=False)
        medianIdx = len(xList) / 2

        # 根节点赋值
        root.value = xList[medianIdx]
        root.layIdx = layerIdx
        layerIdx += 1

        # 建立新的左右子结点
        # 如果 xList 为空则不建立新结点
        if len(xList[:medianIdx]) != 0:
            root.left = kdTree.kdNode()
            root.left.parent = root
            self.treeGrowth(root = root.left,\
                            xList=xList[:medianIdx],\
                            layerIdx=layerIdx,\
                            featureIdx=featureIdx)
        if len(xList[medianIdx+1:]) != 0:
            root.right = kdTree.kdNode()
            root.right.parent = root
            self.treeGrowth(root=root.right,\
                            xList=xList[medianIdx+1:],\
                            layerIdx=layerIdx,\
                            featureIdx=featureIdx)

    def findNearest(self, root, x):
        '''
        传入根结点，找到与 x 最近的实例并返回。
        :param root: 传入的根结点
        :param x: 传入需要计算与之最近的实例点
        :return:
        '''
        pass

    def midTravel(self, root):
        '''
        二叉树的中序遍历。
        :param root: 树节点，首次传入时为树的根节点
        :return:
        '''
        try:
            self.midTravel(root.left)
        except AttributeError as e:
            # print e
            return None
        # 也可以是其他操作
        print("=== root.value:{0} ===".format(root.value))
        try:
            self.midTravel(root.right)
        except AttributeError as e:
            # print e
            return None

def median(xList):
    '''
    计算列表 xList 的中位数并返回。若列表中元素个数为偶数，则中位数为中间两个数的均值，
    如果列表中元素个数为奇数，则中位数为列表中间的元素。
    :param xList: 输入列表 xList
    :return: 返回列表 xList 的中位数
    '''
    if not isinstance(xList, list):
        print("input variable is not list variable.")
        return -1
    xList.sort()
    tmpMidIdx = (len(xList) - 1) / 2
    if len(xList) % 2 == 0:
        midLeftIdx = len(xList) / 2
        midRightIdx = tmpMidIdx
        medianNum = (xList[midLeftIdx] + xList[midRightIdx])/2.0
        return medianNum
    else:
        medianNum = xList[tmpMidIdx]
        return medianNum

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
    print median(xList=[1,2,3])
    print median(xList=[1,2,3,4])
    xx = [1,2]
    print median(xx)

    print("==========================")
    xList = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    tree = kdTree()
    kdTreeRoot = tree.createKDTree(xList=xList)
    tree.midTravel(root=kdTreeRoot)