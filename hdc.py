import math
import os
import numpy as np
import pandas as pd
import pickle
import time
import scipy
import random
'''
HDCFramework要求：
传入的数据为一个列表，列表中的每个元素为一个字符串或一维数字列表
传入的标签为一个列表，列表内的元素为0,1,2...,K-1，K为类数
'''
class HDCFramework:
    def __init__(self, D, N, K, lr):
        self.IM = dict()
        self.AM = dict()
        self.ZAM = dict()
        self.IAM = dict()
        self.D = D # 维数
        self.N = N # ngram的n
        self.K = K # 类数
        self.lr = lr
        self.LangLabels = []
        for key in range(self.K):
            self.AM[key] = np.zeros(self.D)
            self.LangLabels.append(key)
    
    def GenerateHV(self):
        '''
        生成随机向量
        '''
        RandomIndexList = np.random.permutation(self.D)
        RandomHV = np.ones(self.D, dtype=np.int)
        RandomHV[RandomIndexList[0:self.D//2]] = 1
        RandomHV[RandomIndexList[self.D//2:self.D]] = -1
        return RandomHV

    def CalculateCosineSimilarity(self, HV1, HV2):
        '''
        计算余弦相似度
        '''
        DotProduct = np.dot(HV1, HV2.T)
        Norm1 = np.linalg.norm(HV1)
        Norm2 = np.linalg.norm(HV2)
        CosineSimilarity = float(DotProduct) / (Norm1 * Norm2)
        return CosineSimilarity 

    def CalculateHammingSimilarity(self, HV1, HV2):
        '''
        计算余弦相似度
        '''
        # DotProduct = np.dot(HV1, HV2.T)
        # Norm1 = np.linalg.norm(HV1)
        # Norm2 = np.linalg.norm(HV2)
        # CosineSimilarity = float(DotProduct) / (Norm1 * Norm2)
        HammingSimilarity = np.sum(HV1 == HV2)
        return HammingSimilarity 

    def CalculateCustomSimilarity(self, V1, V2, k):
        '''
        对DCT变换后的向量的前k位做相似度计算
        '''
        CustomSimilarity = 0
        for i in range(k):
            CustomSimilarity = CustomSimilarity + abs(V1[i] - V2[i]) * (k - i)
            # CustomSimilarity = CustomSimilarity + abs(V1[i] - V2[i])
        return CustomSimilarity

    def LookupIM(self, Key):
        '''
        在IM中查找Key对应的向量，如果没有则添加向量
        '''
        if not self.IM.__contains__(Key):
            self.IM[Key] = self.GenerateHV();
        return self.IM[Key]

    def NGramEncoding(self, InputSequence):
        '''
        对输入序列中的的若干Ngram编码结果计算并累加
        '''
        block = np.zeros([self.N, self.D])
        sumHV = np.zeros(self.D)
        for i, Key in enumerate(InputSequence):
            block = np.roll(block, (1, 1), (0, 1))
            block[0] = self.LookupIM(Key)
            if i >= self.N - 1:
                NGramHV = block[0]
                for j in range(1, self.N):
                    NGramHV = NGramHV * block[j]
                sumHV = sumHV + NGramHV
        return sumHV
    
    def Binarization(self, HV):
        '''
        对向量进行二值化
        '''
        threshold = 0 # 因为是bipolar向量，所以阈值是0
        for i in range(self.D):
            if HV[i] == 0:
                HV[i] = random.randint(0, 1) * 2 - 1
            elif HV[i] > threshold:
                HV[i] = 1
            else:
                HV[i] = -1
        return HV

    def Train(self, TrainDataList, TrainLabelList):
        length = len(TrainDataList)
        for i in range(length):
            Data = TrainDataList[i]
            Label = TrainLabelList[i]
            tmp = self.NGramEncoding(Data)
            self.AM[Label] = self.AM[Label] + tmp
            # if i % 10000 == 0:
                # print("Train data {0} calculated.".format(str(i)))
        for Label in self.LangLabels:
            self.IAM[Label] = self.AM[Label]
            self.AM[Label] = self.Binarization(self.AM[Label])
    
    def TrainForBoost(self, TrainDataList, TrainLabelList, w):
        length = len(TrainDataList)
        for i in range(length):
            Data = TrainDataList[i]
            Label = TrainLabelList[i]
            tmp = self.NGramEncoding(Data)
            self.AM[Label] = self.AM[Label] + tmp * w[i] # to modify
            # if i % 10000 == 0:
                # print("Train data {0} calculated.".format(str(i)))
        for Label in self.LangLabels:
            self.IAM[Label] = self.AM[Label]
            self.AM[Label] = self.Binarization(self.AM[Label])
    
    def ReTrain(self, ReTrainDataList, ReTrainLabelList):
        for i, ReTrainData in enumerate(ReTrainDataList):
            Data = ReTrainDataList[i]
            Label = ReTrainLabelList[i]
            ReTrainHV = self.Binarization(self.NGramEncoding(Data))
            maxAngle = -1
            predicLang = -1
            for j,LangLabel in enumerate(self.LangLabels):
                angle = self.CalculateHammingSimilarity(self.AM[LangLabel], ReTrainHV)
                if angle > maxAngle:
                    maxAngle = angle
                    predicLang = LangLabel
            if predicLang != Label:
                self.IAM[Label] = self.IAM[Label] + ReTrainHV * self.lr
                self.IAM[predicLang] = self.IAM[predicLang] - ReTrainHV * self.lr
            # if i % 10000 == 0:
                # print("ReTrain data {0} calculated.".format(str(i)))
        for Label in self.LangLabels:
            self.AM[Label] = self.Binarization(self.IAM[Label])

    def ReTrainForBoost(self, ReTrainDataList, ReTrainLabelList, w):
        for i, ReTrainData in enumerate(ReTrainDataList):
            Data = ReTrainDataList[i]
            Label = ReTrainLabelList[i]
            ReTrainHV = self.Binarization(self.NGramEncoding(Data))
            maxAngle = -1
            predicLang = -1
            for j,LangLabel in enumerate(self.LangLabels):
                angle = self.CalculateHammingSimilarity(self.AM[LangLabel], ReTrainHV)
                if angle > maxAngle:
                    maxAngle = angle
                    predicLang = LangLabel
            if predicLang != Label:
                self.IAM[Label] = self.IAM[Label] + ReTrainHV * w[i] * self.lr
                self.IAM[predicLang] = self.IAM[predicLang] - ReTrainHV * w[i] * self.lr
            # if i % 10000 == 0:
                # print("ReTrain data {0} calculated.".format(str(i)))
        for Label in self.LangLabels:
            self.AM[Label] = self.Binarization(self.IAM[Label])

    def ZipHV(self, HV):
        return scipy.fft.dct(x=HV)

    def ZipAM(self):
        for i, Label in enumerate(self.LangLabels):
            self.ZAM[Label] = self.ZipHV(self.AM[Label])

    def Test(self, TestDataList):
        length = len(TestDataList)
        predict = []
        for i in range(length):
            Data = TestDataList[i]
            TestHV = self.Binarization(self.NGramEncoding(Data))
            maxAngle = -1
            predicLang = -1
            for j,LangLabel in enumerate(self.LangLabels):
                angle = self.CalculateHammingSimilarity(self.AM[LangLabel], TestHV)
                if angle > maxAngle:
                    maxAngle = angle
                    predicLang = LangLabel
            predict.append(predicLang)
            # if i % 10000 == 0:
                # print("Test data {0} calculated.".format(str(i)))
        return predict
