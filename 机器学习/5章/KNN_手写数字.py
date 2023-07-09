import numpy as np
from os import listdir


def img2vector(filename, h, w):
    imgVector = np.zeros((1, h * w))
    with open(filename) as fileIn:
        for row in range(h):
            line_str = fileIn.readline()
            for col in range(w):
                imgVector[0, row * 32 + col] = int(line_str[col])
    return imgVector


def myKNN(testDigit, trainX, trainY, k):
    numSamples = trainX.shape[0]
    # 1.计算欧式距离
    diff = []
    for n in range(numSamples):
        diff.append(testDigit - trainX[n])
    diff = np.array(diff)
    squaredDiff = diff ** 2
    squaredDiff = np.sum(squaredDiff, axis=1)
    distance = squaredDiff ** 0.5
    # 2.按距离进行排序
    sortedDistIndices = np.argsort(distance)
    classCount = {}
    for i in range(k):
        # 3.按顺序读取标签
        voteLabel = trainY[sortedDistIndices[i]]
        # 4.计算该标签出现次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 5.查找出现次数最多的类别，作为分类结果
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex


def loadDataSet():
    # 获取训练数据
    print('1.Loading trainSet......')
    trainFileList = listdir('E:/pythonProject/机器学习/datas/trainSet')
    trainNum = len(trainFileList)  # 1068
    trainX = np.zeros((trainNum, 32 * 32))  # [1068, 1024] 的矩阵
    trainY = []
    for i in range(trainNum):
        trainFile = trainFileList[i]
        # img2vector() 将txt中的数据由32*32转化为1*1024的
        # 再将imgVector存入i行中
        trainX[i, :] = img2vector('E:/pythonProject/机器学习/datas/trainSet/%s' % trainFile, 32, 32)
        label = int(trainFile.split('_')[0])
        trainY.append(label)

    # 获取训练数据
    print('12.Loading testSet......')
    testFileList = listdir('E:/pythonProject/机器学习/datas/testSet')
    testNum = len(testFileList)  # 1068
    testX = np.zeros((testNum, 32 * 32))  # [1068, 1024] 的矩阵
    testY = []
    for i in range(testNum):
        testFile = testFileList[i]
        # img2vector() 将txt中的数据由32*32转化为1*1024的
        # 再将imgVector存入i行中
        testX[i, :] = img2vector('E:/pythonProject/机器学习/datas/testSet/%s' % testFile, 32, 32)
        label = int(testFile.split('_')[0])
        testY.append(label)
    return trainX, trainY, testX, testY


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = loadDataSet()
    numTestSamples = test_X.shape[0]
    matchCount = 0
    print('3.Find the most frequent label in k-nearest... And Show the reuslt...')
    for i in range(numTestSamples):
        predict = myKNN(test_X[i], train_X, train_Y, 3)
        print(f'result is: {predict},   real answer:{test_Y[i]}')
        if predict == test_X[i]:
            matchCount = matchCount + 1
    accuracy = float(matchCount)/ numTestSamples
