import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pickle

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
    
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None]*5
    
    def drop_columns(self, kolom):
        self.input_data = self.input_data.drop(columns=kolom)

    def outlierCheck(self, kolom):
        boxplot = self.x_train.boxplot(column=(kolom))
        plt.title('Boxplot of Training Set')
        plt.show()
        boxplot2 = self.x_test.boxplot(column=(kolom))
        plt.title('Boxplot of Testing Set')
        plt.show()

    def createModel(self, criteria='entropy',maxDepth=8):
        self.model = RandomForestClassifier(criterion=criteria, max_depth=maxDepth)
    
    def label_encode_columns(self, columns):
        for col in columns:
            label_encoder = LabelEncoder()
            self.x_train[col] = label_encoder.fit_transform(self.x_train[col])
            self.x_test[col] = label_encoder.fit_transform(self.x_test[col])
    
    def createMeanFromColumn(self,kolom):
        return np.mean(self.x_train[kolom])
    
    #def fillingNAWithNumbers(self, columns):
        #mean = self.x_train[columns].mean()
        #self.x_train[columns] = self.x_train[columns].fillna(mean)
        #self.x_test[columns] = self.x_test[columns].fillna(mean)
    
    def fillingNAWithNumbers(self,columns,number):
        self.x_train[columns].fillna(number, inplace=True)
        self.x_test[columns].fillna(number, inplace=True)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test)

    def createReport(self):
        print("\nClassification Report\n")
        print(classification_report(self.y_test, self.y_predict, target_names=['0', '1']))

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_data, self.output_data, test_size = test_size, random_state = random_state)
    
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def accuracy_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def precision_model(self):
        predictions = self.model.predict(self.x_test)
        return precision_score(self.y_test, predictions)

    def recall_model(self):
        predictions = self.model.predict(self.x_test)
        return recall_score(self.y_test, predictions)

    def f1_model(self):
        predictions = self.model.predict(self.x_test)
        return f1_score(self.y_test, predictions)
    
    def print_input(self):
        print(self.x_train)

    def checkNA_train(self):
        return self.x_train.isna().sum()

    def checkNA_test(self):
        return self.x_test.isna().sum()
    
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

file_path = 'data_D.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('churn')

input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.drop_columns(['Unnamed: 0', 'id', 'CustomerId', 'Surname'])
model_handler.split_data()
model_handler.checkNA_train()
model_handler.checkNA_test()
#model_handler.fillingNAWithNumbers(['CreditScore'])

replace_na = model_handler.createMeanFromColumn('CreditScore')
model_handler.fillingNAWithNumbers('CreditScore',replace_na)

#model_handler.outlierCheck(['CreditScore'])
#model_handler.outlierCheck(['Age'])
#model_handler.outlierCheck(['Tenure'])
#model_handler.outlierCheck(['Balance'])
#model_handler.outlierCheck(['EstimatedSalary'])

model_handler.label_encode_columns(['Geography', 'Gender'])
model_handler.print_input()

model_handler.createModel()
model_handler.train_model()
model_handler.makePrediction()

print("Model Accuracy:", model_handler.accuracy_model())
print("Model Precision:", model_handler.precision_model())
print("Model Recall:", model_handler.recall_model())
print("Model F1 Score:", model_handler.f1_model())
model_handler.createReport()

model_handler.save_model_to_file('OOPmodel4.pkl') 
