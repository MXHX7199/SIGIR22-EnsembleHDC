import numpy as np
import scipy as sp
from hdc import HDCFramework
from sklearn.metrics import accuracy_score

class AdaBoostFrameWork:
	'''
	K 类别数
	M 分类器个数
	LearningRateForBoost 学习率
	[G1, G2, ..., GM] 分类器列表
	[a1, a2, ..., aM] 分类器系数数组

	DimForHDC HDC的维数
	NForNgram HDC中Ngram编码的N的值
	LearningRateForClassifier 分类器的学习率
	'''

	def __init__(self, K, M, LearningRateForBoost, DimForHDC, NForNgram, LearningRateForHDC):
		self.M = M
		self.K = K
		self.learningrate = LearningRateForBoost
		self.G = [HDCFramework(10000, NForNgram, self.K, LearningRateForHDC) for _ in range(self.M)]
		self.a = [0 for _ in range(self.M)]

	def Train(self, TrainData, TrainLabel, TestData, TestLabel, RetrainNumForHDC):
		length = len(TrainData)
		w = np.array([float(1/length) for _ in range(length)])
		# Step 1
		for i in range(length):
			w[i] = 1 / length # to modify 这里设置这么小的权值还有待商榷
		# step 2
		y = np.array(TrainLabel)
		for i in range(self.M):
			print("■■■■■■■■■■■■■■■■■■■■ HDC{} ■■■■■■■■■■■■■■■■■■■■".format(str(i)))
			print("-------------- Begin train HDC{} --------------".format(str(i)))
			# step 2.a
			self.G[i].TrainForBoost(TrainData, TrainLabel, w)
			for j in range(RetrainNumForHDC):
				self.G[i].ReTrainForBoost(TrainData, TrainLabel, w)
			print("--------------- End train HDC{} ---------------".format(str(i)))
			# extra
			if i < 1:
				pred = self.G[i].Test(TestData)
				acc = accuracy_score(TestLabel, pred)
				print("extra test for HDC model in round {0}: acc={1}".format(str(i), str(acc)))

			# todo retrain
			# step 2.b
			print("-------- Begin Test HDC{} on traindata --------".format(str(i)))
			y_predict = np.array(self.G[i].Test(TrainData))
			print("--------- End Test HDC{} on traindata ---------\n".format(str(i)))
			incorrect = y_predict != y
			e = np.mean(np.average(incorrect, weights=w, axis=0))
			if e <= 0:
				print("Early Terminated!")
				self.a[i] = 1
				e = 0
			else:
				if e >= 1.0 - (1.0 / self.K):
					raise ValueError(
						"BaseClassifier in AdaBoostClassifier "
						"ensemble is worse than random, ensemble "
						"can not be fit."
					)
				# step 2.c
				self.a[i] = self.learningrate * (
					np.log((1.0 - e) / e) + np.log(self.K - 1.0)
				)

				# step 2.d
				if i < self.M - 1:
					w = np.exp(
					np.log(w)
					+ self.a[i] * incorrect * (w > 0)
				)
			if e == 0:
				break
			wsum = np.sum(w)
			if not np.isfinite(wsum):
				print("Sample weights have reached infinite values, at iteration {}, causing overflow. Iterations stopped. Try lowering the learning rate.".format(str(i)))
				break
			if wsum <= 0:
				break
			if i < self.M - 1:
				w = w / wsum

	def Test(self, TestData, TestLabel):
		print("\n--------------- Begin Final Test ---------------")
		classes = np.array([i for i in range(self.K)]) # 这里不要+1 对其label 0,1,2...21
		classes = classes[:, np.newaxis]
		pred = sum(
			(np.array(self.G[i].Test(TestData)) == classes).T * self.a[i] for i in range(self.M)
		)
		pred /= np.array(self.a).sum()
		print("---------------- End Final Test ----------------")
		return pred