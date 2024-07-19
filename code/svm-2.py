# import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import svm_file
import Statistical_Analysis

# about csv data
class DiabetesAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data, self.diabetes_patients, self.non_diabetes_patients = self.load_data()

    # load data 
    def load_data(self):
        data = pd.read_csv(self.file_path)
        diabetes_patients = data[data['Outcome'] == 1]
        non_diabetes_patients = data[data['Outcome'] == 0]
        return data, diabetes_patients, non_diabetes_patients
    
    # feature selection based on correlation
    def feature_extraction_analysis(self):
        columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        stat = Statistical_Analysis.Stat_Analysis(columns)
        stat.Diabetes_Boxplot(self.diabetes_patients, self.non_diabetes_patients)
        stat.Diabetes_Correlation_Heatmap(self.data)
        stat.Diabetes_histogram_pdf(self.diabetes_patients, self.non_diabetes_patients)

    # preprocessing for training
    def preprocess_for_svm(self, feature_columns):
        feature_vector = self.data[feature_columns].values
        objective_vector = self.data['Outcome'].values
        return train_test_split(feature_vector, objective_vector, test_size=0.3, random_state=1)

def main():
    file_path = 'PATH'
    analysis = DiabetesAnalysis(file_path)
    
    dataset = analysis.preprocess_for_svm(['Glucose', 'BMI'])
    
    # training
    svm_file.Problem_1().SVM_support_vector_machine(dataset, Result='Test', Kernel_Mode='linear')

if __name__ == "__main__":
    main()