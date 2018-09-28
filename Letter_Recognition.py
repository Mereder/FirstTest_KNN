import numpy as np
import operator




def file2matrix(filename):
    fileRead = open(filename,'r',encoding='utf-8')
    #读取文件所有内容
    arrayAllLines = fileRead.readlines()
    arrayAllLines[0] = arrayAllLines[0].lstrip('\ufeff')
    #取文件行数
    numberOfLines = len(arrayAllLines)
    #生成Numpy 的矩阵 zeros 生成矩阵
    #Letter 识别 第一列为字母 剩下 2-17列为 16个特征值
    returnMat = np.zeros((numberOfLines,16))
    #标签向量
    letterLabelVector = []
    # 行索引
    index = 0
    #处理每行数据
    for line in arrayAllLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # split 根据特定分隔符进行切片
        listFromLine = line.split(',')
        # 特征矩阵中 存放1-16的特征值
        returnMat[index,:] = listFromLine[1:17]
        # 将字母标记进行分类
        if listFromLine[0] == 'A':
            letterLabelVector.append('A')
        elif listFromLine[0] == 'B':
            letterLabelVector.append('B')
        elif listFromLine[0] == 'C':
            letterLabelVector.append('C')
        elif listFromLine[0] == 'D':
            letterLabelVector.append('D')
        elif listFromLine[0] == 'E':
            letterLabelVector.append('E')
        elif listFromLine[0] == 'F':
            letterLabelVector.append('F')
        elif listFromLine[0] == 'G':
            letterLabelVector.append('G')
        elif listFromLine[0] == 'H':
            letterLabelVector.append('H')
        elif listFromLine[0] == 'I':
            letterLabelVector.append('I')
        elif listFromLine[0] == 'J':
            letterLabelVector.append('J')
        elif listFromLine[0] == 'K':
            letterLabelVector.append('K')
        elif listFromLine[0] == 'L':
            letterLabelVector.append('L')
        elif listFromLine[0] == 'M':
            letterLabelVector.append('M')
        elif listFromLine[0] == 'N':
            letterLabelVector.append('N')
        elif listFromLine[0] == 'O':
            letterLabelVector.append('O')
        elif listFromLine[0] == 'P':
            letterLabelVector.append('P')
        elif listFromLine[0] == 'Q':
            letterLabelVector.append('Q')
        elif listFromLine[0] == 'R':
            letterLabelVector.append('R')
        elif listFromLine[0] == 'S':
            letterLabelVector.append('S')
        elif listFromLine[0] == 'T':
            letterLabelVector.append('T')
        elif listFromLine[0] == 'U':
            letterLabelVector.append('U')
        elif listFromLine[0] == 'V':
            letterLabelVector.append('V')
        elif listFromLine[0] == 'W':
            letterLabelVector.append('W')
        elif listFromLine[0] == 'X':
            letterLabelVector.append('X')
        elif listFromLine[0] == 'Y':
            letterLabelVector.append('Y')
        elif listFromLine[0] == 'Z':
            letterLabelVector.append('Z')
        index += 1
    return returnMat, letterLabelVector
    pass


def autoNorm(letterRecog):
    # 获取单列数据最小值
    minVals = letterRecog.min(0)
    # 获取单列数据最大值
    maxVals = letterRecog.max(0)
    # 取值范围  也是作为分母
    ranges = maxVals - minVals
    # shape(矩阵)  返回矩阵的行列数
    # 创建一个初始化矩阵 与 letterRecog 同样大小
    normalMat = np.zeros(np.shape(letterRecog))
    # 返回行数
    m = letterRecog.shape[0]
    # 将各列最小的数据 向下拓宽M行 与 原来矩阵同样大小 进行减法操作
    normalMat = letterRecog - np.tile(minVals,(m,1))
    # 进行归一化
    normalMat = normalMat / np.tile(ranges,(m,1))

    return normalMat, ranges, minVals


def Rrecognize(Test, DataSet, letterLabels, k):
    DataSetSize = DataSet.shape[0]
    # 计算过程 先将 测试集拓展为 DataSet 大小后 进行做差  乘方 相加  开方
    # 拓展测试集 并 做差
    diffMal = np.tile(Test,(DataSetSize,1)) - DataSet
    # 乘方
    powDiffMal = diffMal**2
    # 行相加
    # sum()所有元素相加,sum(0)列相加,sum(1)行相加
    # 一行元素进行相加 所得结果 放入一维矩阵  共 Size个数据
    sumDistance = powDiffMal.sum(1)
    # 开方计算出距离
    disTance = sumDistance**0.5
    # 将距离进行排序，距离小的 数据的 索引在前面
    # 注意 排序得到的是索引
    sortedDistanceIndex = disTance.argsort()
    # 定义一个 记录各字母次数的词典
    # 字典的每个键值 key=>value 对用冒号 : 分割，每个键值对之间用逗号 , 分割，整个字典包括在花括号 {} 中
    letterCount = {}
    # 取前k个
    for i in range(k):
        voteLabel = letterLabels[sortedDistanceIndex[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 统计字母的次数
        letterCount[voteLabel] = letterCount.get(voteLabel,0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedLetterCount = sorted(letterCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedLetterCount)
    return sortedLetterCount[0][0]


def LetterRecognitionTest():
    filename = "letter-recognition.data"
    #读取文件，返回的特征矩阵和分类向量
    letterRecog, letterLabels = file2matrix(filename)
    #取 所有数据的 比值
    hoRatio = 0.10
    # 数据归一化处理 返回归一化矩阵  数据范围 数据最小值
    normalMat, ranges, minVals = autoNorm(letterRecog)
    # 获得行数
    m = normalMat.shape[0]
    # 测试数据个数
    numTestVecs = 50
    # 错误分类的情况计数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 暂时取50个作为测试集
        recognitionResult =Rrecognize(normalMat[i,:], normalMat[numTestVecs:m,:],
                                      letterLabels[numTestVecs:m], 5)
        print("分类字母:%s\t真实字母:%s" % (recognitionResult, letterLabels[i]))
        if recognitionResult != letterLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))

if __name__ == '__main__':
    LetterRecognitionTest()