import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score





class HeartPrediction(LogisticRegression):

    def __init__(self, file:str, target:str, max_iter=500):

        self.data = pd.read_csv(file)

        
        self.target = target

        self.x = np.array(self.data.drop([target], axis=1))

        self.y = np.array(self.data[target])

        self.x_train = None
        self.y_train = None
        self.x_test = None


        super().__init__(max_iter=max_iter)



    def train_test_split(self, test_split_size:float=0.1):
        self.x_train, self.x_test, self.y_train, self.y_test =model_selection.train_test_split(self.x, self.y, test_size=test_split_size)
        
    
    def fit(self, sample_weight=None):
    
        return super().fit(self.x_train, self.y_train, sample_weight)


    def predictions(self):

        return super().predict(self.x_test)


    def accuracy(self):
        return accuracy_score(self.y_test, self.predictions())



model = HeartPrediction("heart_disease.csv", "target", max_iter=1000)


model.train_test_split(0.1)

model.fit()

print(model.accuracy())








