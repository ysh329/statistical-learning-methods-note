# -*- coding: utf-8 -*-
################################### PART0 DESCRIPTION #################################
# Filename: Simple-kd-Tree.py
# Description: kd 树构造的简化版，有构造的简化实现。
# 此外，还有与输入 x 同一维度特征空间下计算近似最
# 近邻的实现。

# E-mail: ysh329@sina.com
# Create: 2016-12-05 17:01:52
# Last:
__author__ = 'yuens'


################################### PART1 IMPORT ######################################


################################### PART2 CLASS && FUNCTION ###########################
class Node(object):
    '''
    构造结点并生成 kd 树，kd 树是一种二叉树，构造方法与一般的二叉树差别不大，
    每个结点根据当前样本集中的当前特征中位数作为划分为两部分，两部分即左右子
    结点。递归生成子结点，直到样本集中没有实例，递归完也就构造完了 kd 树。
    '''
    def __init__(self, xList, layerIdx=0, featureIdx=0, parent=None):
        '''
        基于列表样本集数据 xList 构造 kd 二叉树，根据当前样本集中当前特征构造。
        注意：每个实例样本都是一个 list 类型，换句话说 xList 是一个二维 list 。
        :param xList: 包含所有样本的二维 list
        :param layerIdx: 要构造的层数（由 1 开始）
        :param featureIdx: 要构造的结点的 value 选取的参照是第 featureIdx 个特征
                            featureIdx 从 0 开始
        :param parent: 要构造的结点的父结点（用于递归向上查找近邻结点），考虑到
        第一个根节点没有父结点，所以默认为 None
        '''
        # 当前分支样本遍历完
        if len(xList)==0:
            return None
        # 按照特征下标 featureIdx 进行排序找中位数下标
        if featureIdx == len(xList[0]):
            featureIdx = 0
        self.featureIdx = featureIdx
        featureIdx += 1
        self.parent = parent
        xList.sort(key=lambda x: x[self.featureIdx],\
                   reverse=False)
        medianIdx = len(xList) / 2
        self.value = xList[medianIdx]
        self.layerIdx = layerIdx
        layerIdx += 1
        # 建立新的左右子结点
        self.left = None if len(xList)<=1 else Node(xList=xList[:medianIdx],\
                                                    layerIdx=layerIdx,\
                                                    featureIdx=featureIdx,\
                                                    parent=self)
        self.right = None if len(xList)<=1 else Node(xList=xList[medianIdx+1:],\
                                                     layerIdx=layerIdx,\
                                                     featureIdx=featureIdx,\
                                                     parent=self)

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
            return root.value
        elif x[featureIdx] < root.value[featureIdx] and root.left != None:
            featureIdx += 1
            if featureIdx == len(x):
                featureIdx = 0
            return self.findApproxNearestInSameDim(root.left, x, featureIdx)
        elif x[featureIdx] < root.value[featureIdx] and root.left == None:
            return root.value
        elif root.value[featureIdx] < x[featureIdx] and root.right != None:
            featureIdx += 1
            if featureIdx == len(x):
                featureIdx = 0
            return self.findApproxNearestInSameDim(root.right, x, featureIdx)
        elif root.value[featureIdx] < x[featureIdx] and root.right == None:
            return root.value
        else:
            print("find Nearest Unexpected Error")
            return None

################################### PART3 TEST ########################################
# 例子
if __name__ == "__main__":
    xList = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]

    print("====== create instance ======")
    root = Node(xList=xList)
    print("root.value:{0}".format(root.value))

    print("====== midTravel ======")
    root.midTravel(root=root)

    print("====== findApproxNearest ======")
    x = [9, 6]
    approxNearestX = root.findApproxNearestInSameDim(root, x)
    print("approxNearestX:{0}".format(approxNearestX))