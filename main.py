import numpy as np
import pandas as pd
import os
import glob
from hdc import HDCFramework
from adaboost import AdaBoostFrameWork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import argparse

# 语言分类数据集 22类 22万 + 2万
def LoadDataset1(datarate):
    TrainData = []
    TrainLabel = []
    TestData = []
    TestLabel = []
    LangLabels = [
            'afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav', 'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe'
        ]
    for i, Label in enumerate(LangLabels):
        FilePath = os.path.join("language", "training_texts", "{}.txt".format(Label))
        with open(FilePath,'r',encoding='utf-8') as f:
            InputSequence = f.read()
        senlist = InputSequence.split('\n')
        for data in senlist:
            TrainData.append(data)
            TrainLabel.append(i)
    print("{} sentences have been loaded for train.".format(str(len(TrainData))))

    ShortLangLabels = [
            'af', 'bg', 'cs', 'da', 'nl', 'de', 'en', 'et', 'fi', 'fr', 'el', 'hu', 'it', 'lv', 'lt', 'pl', 'pt', 'ro', 'sk', 'sl', 'es', 'sv'
    ]
    mp = dict()
    for i,key in enumerate(ShortLangLabels):
        mp[key] = i
    TestFilePathList = glob.glob(os.path.join("language", "testing_texts", "*.txt"))
    for i, TestFilePath in enumerate(TestFilePathList):
        (_, TestFileName) = os.path.split(TestFilePath)
        if not mp.__contains__(TestFileName[0:2]):
            continue
        Label = mp[TestFileName[0:2]]
        with open(TestFilePath,'r',encoding='utf-8') as f:
            InputSequence = f.read()
        TestData.append(InputSequence[:-1]) # 去掉回车，有待验证是不是所有文本都有
        TestLabel.append(Label)
    print("{} sentences have been loaded for test.".format(str(len(TestData))))

    # _, TrainData, _, TrainLabel = train_test_split(TrainData, TrainLabel, test_size=datarate, stratify=TrainLabel, random_state=1)
    # _, TestData, _, TestLabel = train_test_split(TestData, TestLabel, test_size=datarate, stratify=TestLabel, random_state=1)
    return 22, TrainData, TrainLabel, TestData, TestLabel

# 情感分类数据集 2类
def LoadDataset2(datarate):
    TrainData = []
    TrainLabel = []
    TestData = []
    TestLabel = []
    TrainFileName = os.path.join("SST-2", "train.tsv")
    frame = pd.read_csv(TrainFileName, sep='\t')
    TrainDataList = frame.values.tolist()
    for i, data in enumerate(TrainDataList):
        Label = data[1]
        TrainLabel.append(Label)
        TrainData.append(data[0])
    print("{} sentences have been loaded for train.".format(str(len(TrainData))))
    
    TestFileName = os.path.join("SST-2", "dev.tsv")
    frame = pd.read_csv(TestFileName, sep='\t')
    TestDataList = frame.values.tolist()
    for i, data in enumerate(TestDataList):
        Label = data[1]
        TestLabel.append(Label)
        TestData.append(data[0])
    print("{} sentences have been loaded for test.".format(str(len(TestData))))
    # _, TrainData, _, TrainLabel = train_test_split(TrainData, TrainLabel, test_size=datarate, stratify=TrainLabel, random_state=1)
    # _, TestData, _, TestLabel = train_test_split(TestData, TestLabel, test_size=datarate, stratify=TestLabel, random_state=1)
    return 2, TrainData, TrainLabel, TestData, TestLabel

# 新闻分类数据集 4类 12万
def LoadDataset3(datarate):
    TrainData = []
    TrainLabel = []
    TestData = []
    TestLabel = []

    # ctrl = 100 # to modify
    # count = [0 for _ in range(4)] # to modify

    TrainFileName = os.path.join("ag_news_csv", "train.csv")
    frame = pd.read_csv(TrainFileName, header=None)
    TrainDataList = frame.values.tolist()
    for i, data in enumerate(TrainDataList):
        Label = data[0] - 1 # 注意，这个数据集的类编号从1开始，因此要减1
        # count[Label] = count[Label] + 1
        # if count[Label] > ctrl: # to modify
        #     continue
        TrainLabel.append(Label)
        TrainData.append(data[1] + data[2])
    print("{} sentences have been loaded for train.".format(str(len(TrainData))))
    # count = [0 for _ in range(4)] # to modify
    
    TestFileName = os.path.join("ag_news_csv", "test.csv")
    frame = pd.read_csv(TestFileName, header=None)
    TestDataList = frame.values.tolist()
    for i, data in enumerate(TestDataList):
        Label = data[0] - 1
        # count[Label] = count[Label] + 1
        # if count[Label] > ctrl // 4: # to modify
        #     continue
        TestLabel.append(Label)
        TestData.append(data[1] + data[2])
    print("{} sentences have been loaded for test.".format(str(len(TestData))))
    # _, TrainData, _, TrainLabel = train_test_split(TrainData, TrainLabel, test_size=datarate, stratify=TrainLabel, random_state=1)
    # _, TestData, _, TestLabel = train_test_split(TestData, TestLabel, test_size=datarate, stratify=TestLabel, random_state=1)
    return 4, TrainData, TrainLabel, TestData, TestLabel

# 垃圾邮件数据集 2 类
def LoadDataset4(datarate):
    df = pd.read_csv("spam.csv", delimiter=',', encoding='latin-1')
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    X = df.v2
    Y = df.v1
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    X_ = []
    for item in X:
        token_item = []
        for letter in item.lower():
            #print(letter)
            if ord(letter) >= ord('a') and ord(letter) <= ord('z'):
                token_item.append(ord(letter) - ord('a') + 11)
            elif ord(letter) >= ord('0') and ord(letter) <= ord('9'):
                token_item.append(ord(letter) - ord('0') + 1)
            else:
                token_item.append(0)
        X_.append(token_item)
    TrainData, TestData, TrainLabel, TestLabel = train_test_split(X_, Y, test_size = 0.2, random_state = 556)
    return 2, TrainData, TrainLabel, TestData, TestLabel

# youtube垃圾邮件数据集 2类
def LoadDataset5(datarate):
    df = pd.read_csv("Youtube-all.csv", delimiter=',', encoding='latin-1')
    X = df.CONTENT
    Y = df.CLASS
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    X_ = []
    for item in X:
        token_item = []
        for letter in item.lower():
            #print(letter)
            if ord(letter) >= ord('a') and ord(letter) <= ord('z'):
                token_item.append(ord(letter) - ord('a') + 11)
            elif ord(letter) >= ord('0') and ord(letter) <= ord('9'):
                token_item.append(ord(letter) - ord('0') + 1)
            else:
                token_item.append(0)
        X_.append(token_item)
    TrainData, TestData, TrainLabel, TestLabel = train_test_split(X_, Y, test_size=0.1, random_state=19720)
    downsample = None
    TrainData = TrainData[:downsample]
    TrainLabel = TrainLabel[:downsample]
    return 2, TrainData, TrainLabel, TestData, TestLabel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='--------Begin instructions--------',
        epilog='---------End instructions---------'
    )
    parser.add_argument("--task-id", type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--classifiers", type=int, default=4)
    parser.add_argument("--boost-lr", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=10000)
    parser.add_argument("--ngram", type=int, default=4)
    parser.add_argument("--retrain-rounds", type=int, default=1)
    parser.add_argument("--hdc-lr", type=float, default=0.0005)

    # args = parser.parse_args("--task-id 5 --classifiers 4 --boost-lr 1.0 --dim 2000 --ngram 4 --retrain-rounds 0 --hdc-lr 0.0005".split())
    # python main.py --task-id 5 --classifiers 4 --boost-lr 1.0 --dim 2000 --ngram 4 --retrain-rounds 0 --hdc-lr 0.0005
    args = parser.parse_args()

    funcidx = args.task_id
    M = args.classifiers
    LearningRateForBoost = args.boost_lr
    DimForHDC = args.dim
    NForNgram = args.ngram
    LearningRateForHDC = args.hdc_lr
    RetrainNumForHDC = args.retrain_rounds

    datarate = 1
    funcname = "LoadDataset{}".format(str(funcidx))
    print("Loading Dataset {}".format(str(funcidx)))
    K, TrainData, TrainLabel, TestData, TestLabel = globals().get(funcname)(datarate=datarate)
    print("Dataset {0} loaded, {1} classes in this dataset\n".format(str(funcidx), str(K)))

    ''' origin parameter
        M = 5
        LearningRateForBoost = 1.0 # 0.1 for task5
        DimForHDC = 2000
        NForNgram = 4
        LearningRateForHDC = 0.0005 # 【retrain的学习率】
        RetrainNumForHDC = 0 # 【retrain轮数，这里先不用】
    '''

    AdaBoost = AdaBoostFrameWork(K=K, M=M, LearningRateForBoost=LearningRateForBoost, DimForHDC=DimForHDC, NForNgram=NForNgram, LearningRateForHDC=LearningRateForHDC)

    AdaBoost.Train(TrainData=TrainData, TrainLabel=TrainLabel, TestData=TestData, TestLabel=TestLabel, RetrainNumForHDC=RetrainNumForHDC)

    pred = AdaBoost.Test(TestData=TestData, TestLabel=TestLabel)
    tot = len(TestData)
    acc = 0
    for i in range(tot):
        maxp = 0
        predictlabel = 0
        for j in range(K):
            if maxp < pred[i][j]:
                maxp = pred[i][j]
                predictlabel = j
        if predictlabel == TestLabel[i]:
            acc = acc + 1
    print("■■■■■■■■■■■■■■■ Final acc: {} ■■■■■■■■■■■■■■■".format(str(acc/tot)))

