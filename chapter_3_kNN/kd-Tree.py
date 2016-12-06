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
import math

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
        kd 树的二叉结点。由于搜索 kd 树时候需要回退，所以也要构建出结点的
        父结点，即 parent 。
        '''
        def __init__(self, value=None, left=None, right=None, parent=None, featureIdx=None, layerIdx=None):
            self.value = value
            self.featureIdx = featureIdx
            self.layerIdx = layerIdx

            self.left = left
            self.right = right
            self.parent = parent

    def __init__(self, sampleNum, featureNum, k=None, lengthP=None):
        '''
        初始化模型参数
        :param sampleNum: 训练集样本个数
        :param featureNum: 每个样本的特征个数
        :param k: 分类基于最近的 k 个样本
        :param lengthP: $L_p$ 距离参数
        '''
        # 参数检查
        # 如果为空则计算最近邻
        if k == None:
            k = 1
        # 如果为空则计算欧氏距离
        if lengthP == None:
            lengthP = 2

        self.sampleNum = sampleNum
        self.featureNum = featureNum
        self.k = k
        self.p = float(lengthP)

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

    def findNearest(self, root, x, bestDist, approxNearestRoot=None, isFirstFind=True):
        '''
        传入根结点，找到与 x 最近的实例并返回。
        :param root: 传入的根结点
        :param x: 传入需要计算与之最近的实例点
        :return:
        '''
        if isFirstFind:
            approxNearestX, approxNearestRoot = self.findApproxNearestInSameDim(root=root,\
                                                                                x=x)
            bestDist = self.distanceBetween(x, approxNearestRoot.value)
            if approxNearestRoot.parent != None:
                return self.findNearest(root=approxNearestRoot.parent,\
                                        x=x,\
                                        bestDist=bestDist,\
                                        approxNearestRoot=approxNearestRoot,\
                                        isFirstFind=False)
            else:
                return
        else:
            tmpDist = self.distanceBetween(x, root.value)
            if tmpDist < bestDist:
                approxNearestRoot = root




    def findApproxNearestInSameDim(self, root, x, featureIdx=0):
        '''
        在由树根结点 root 产生的结点中找到输入样本 x 同一
        维度空间的近似最近邻。
        :param root: 二叉树结点，首次传入时为树的根节点
        :param x: 输入样本 x
        :param featureIdx: 当期待比较的特征下标（从 0 开始）
        :return:
        '''
        if root.value == x:
            return root.value, root
        elif x[featureIdx] < root.value[featureIdx] and root.left != None:
            featureIdx += 1
            if featureIdx == len(x):
                featureIdx = 0
            return self.findApproxNearestInSameDim(root.left, x, featureIdx)
        elif x[featureIdx] < root.value[featureIdx] and root.left == None:
            return root.value, root
        elif root.value[featureIdx] < x[featureIdx] and root.right != None:
            featureIdx += 1
            if featureIdx == len(x):
                featureIdx = 0
            return self.findApproxNearestInSameDim(root.right, x, featureIdx)
        elif root.value[featureIdx] < x[featureIdx] and root.right == None:
            return root.value, root
        else:
            print("find Nearest Unexpected Error")
            return None

    def distanceBetween(self, aList, bList, p=None):
        '''
        计算两个点，表示为 aList 与 bList，二者之间的 $L_p$ 距离。
        :param aList: 第一个实例样本的特征
        :param bList: 第二个实例样本的特征
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
        print("root.value:{0}".format(root.value))
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
    print("=== read data ===")
    idList, xList, yList = readDataFrom(path=dataPath,\
                                        hasHeader=hasHeader)
    print("idList:{0}".format(idList))
    print("xList:{0}".format(xList))
    print("yList:{0}".format(yList))
    print

    # 实际采用的数据：数据来自书上的例题
    print("=== unlabeled data ===")
    xList = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    print("xList:{0}".format(xList))
    print

    # 实例化kd-Tree
    print("=== create kd-Tree ===")
    tree = kdTree(sampleNum=len(xList), featureNum=len(xList[0]), k=1, lengthP=2)
    kdTreeRoot = tree.createKDTree(xList=xList)
    print("midTravel order: ")
    tree.midTravel(root=kdTreeRoot)
    print

    # 查找与 x 最近的实例
    '''
    x 可以是已经在训练集中的样本，
    也可是新数据样本。
    '''
    print("=== Find Nearest Instance ===")
    x = [9, 6]
    print("x:{0}".format(x))
    approxNearestX, approxNearestRoot = tree.findApproxNearestInSameDim(kdTreeRoot, x)
    print("approxNearestX:{0}".format(approxNearestX))
    print("approxNearestRoot:{0}".format(approxNearestRoot))
