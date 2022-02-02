import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import svm_file
import Statistical_Analysis

class Problem_2:
        
    def Load_data (self, File): # load_data
        PATH = 'C:/Users/user/Desktop/SVM_Project/'
        DATA = pd.read_csv(PATH + File)    
        Diabetes_Outcome,Non_Diabetes_Outcome=[ DATA['Outcome'] == 1, DATA['Outcome'] == 0]
        Diabetes_Patient = DATA[Diabetes_Outcome] # Diabetes_patients
        Non_Diabetes_Patient = DATA[Non_Diabetes_Outcome] # Non_diabetes_patients
        return DATA, Diabetes_Patient, Non_Diabetes_Patient 

    def Feature_Extraction_Analysis(self, DATA, Diabetes_Patient, Non_Diabetes_Patient): # DATA
        DATA_Column = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
        STAT = Statistical_Analysis.Stat_Analysis(DATA_Column) # Call class (Statistical_Analysis)
        STAT.Diabetes_Boxplot(Diabetes_Patient, Non_Diabetes_Patient) # Box Plot
        STAT.Diabetes_Correlation_Heatmap(DATA) # Correlation
        STAT.Diabetes_histogram_pdf(Diabetes_Patient, Non_Diabetes_Patient) # Probability density
        
    def Preprocessing_before_SVM (self, Column_name, DATA): # Preprocessing DATA
        Feature_Vector = np.column_stack((DATA[Column_name[0]], DATA[Column_name[1]])) # shape = (768, 2)
        Objective_Vector = DATA['Outcome'] # Objective_vector 
        X_train, X_test, Y_train, Y_test = train_test_split(Feature_Vector, Objective_Vector, test_size = 0.3, random_state= 1) 
        # Data split + shuffling
        return X_train, X_test, Y_train, Y_test, Feature_Vector, Objective_Vector
        
if __name__ == "__main__" :
    DATA, Diabetes_Patient, Non_Diabetes_Patient = Problem_2().Load_data("diabetes.csv") # load_data
    #Problem_2().Feature_Extraction_Analysis(DATA, Diabetes_Patient, Non_Diabetes_Patient) # Feature_extraction 
    Dataset = Problem_2().Preprocessing_before_SVM(['Glucose', 'BMI'], DATA) # Feature: glucose , bmi 
    #svm_file.Problem_1().Graph_Total_Train_Test(Dataset) # Plotting Train(70%), Test(30%), Total(100%) 
    svm_file.Problem_1().SVM_support_vector_machine(Dataset, Result = 'Test', Kernel_Mode = 'linear')
    
    
    
    