import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd

class Problem_1:

    def Load_data (self, File_list): # load data ... File_list: file_name_list
        Total_x_vector_data, Total_y_vector_data = [], []
        PATH = 'C:/Users/user/Desktop/SVM_Project/'
        for File_Name in File_list: # File list = ["Case1_Class_0.txt", "Case1_Class_1.txt"]
            File = open(PATH+File_Name) # File open 
            lines = File.readlines()
            for line in lines:
                Data = [float(line.split(" ")[0]), float(line.split(" ")[1])] # File read
                Total_x_vector_data.append(Data) # Total Feature vector 
            Total_y_vector_data += [int(File_Name[-5])]* len(lines) # Total objective vector 
        Total_x_vector_data = np.array(Total_x_vector_data) # convert array
        Total_y_vector_data = np.array(Total_y_vector_data) # convert array
        X_train, X_test, Y_train, Y_test = train_test_split(Total_x_vector_data, Total_y_vector_data, 
                                                            test_size = 0.3, random_state = 1) # Data split + shuffling 
        return X_train, X_test, Y_train, Y_test, Total_x_vector_data, Total_y_vector_data
        
    def Graph_Total_Train_Test (self, Dataset): # Plotting Train(70%), Test(30%), Total(100%)
        '''
        Dataset = [Train_feature_vector, Test_feature_vector, Train_objective_vector, Test_objective_vector, 
                   Total_feature_vector, Total_objective_vector]
        '''
        plt.figure(figsize = (12, 4), constrained_layout = True)
        X_Index = [0, 1, 4] # Feature_vector index (Train, Test, Total)
        Y_Index = [2, 3, 5] # Objective_vector index (Train, Test, Total)
        Title = ["Training Data (70%)", "Test Data (30%)", "Data (100%)"]
        Number=1
        for X,Y in zip (X_Index, Y_Index):
            col = []
            for cols in Dataset[Y]:
                if cols == 1: col.append('red') # color separation about shuffled data
                elif cols == 0: col.append('blue') # color separation about shuffled data
            for Index in range(len(Dataset[X])): # auto_subplot 
                plt.subplot(1,3,Number); plt.title(Title[Number-1]) 
                plt.scatter(Dataset[X][Index,0], Dataset[X][Index,1], c = col[Index])
            Number+=1
        plt.tight_layout(); plt.show()
        
    def SVM_support_vector_machine (self, Dataset, Result, Kernel_Mode, **kargs):       
        '''
        Dataset = [Train_feature_vector, Test_feature_vector, Train_objective_vector, Test_objective_vector, 
                   Total_feature_vector, Total_objective_vector]
        Result = 'Test', 'Train'
        Kernel_Mode = 'linear', 'rbf', 'poly' ... 
        '''
        
        if Result== "Train": Index = 0; # Train_result or Test_result 
        elif Result == "Test": Index = 1; 
        param={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} # params 
        classifier = GridSearchCV(SVC(kernel = Kernel_Mode, **kargs), param, refit = True, return_train_score = True)
        classifier.fit(Dataset[4], Dataset[5]) # Total_feature_vector, Total_objective_vector
        scores_df = pd.DataFrame(classifier.cv_results_) # DataFrame-result 
        print(scores_df[['param_C', 'param_gamma', 'mean_train_score', 'mean_test_score', 'rank_test_score']])
        print("optimized parameter :",classifier.best_params_, "\nBest_mean_test_score:",classifier.best_score_)
        #(params, train_score, test_score, rank) + (best_params, scores) 
        
        plt.figure(figsize = (8, 8)) # Plotting Decision boundary 
        col = []
        for cols in Dataset[Index+2]:
            if cols == 1: col.append('red')
            elif cols == 0: col.append('blue') 
        for i in range(len(Dataset[Index])):
            if col==0: plt.scatter(Dataset[Index][i,0], Dataset[Index][i,1], c = col[i])
            else: plt.scatter(Dataset[Index][i,0], Dataset[Index][i,1], c = col[i])
        x=np.linspace(np.min(Dataset[4][:, 0]), np.max(Dataset[4][:, 0]), 30) # x
        y=np.linspace(np.min(Dataset[4][:, 1]), np.max(Dataset[4][:, 1]), 30) # y for Decision boundary
        X,Y=np.meshgrid(x, y) # coordinatesD
        xy=np.vstack([X.ravel(), Y.ravel()]).T # .ravel: Multi-dimension --> One-dimension 
        Z=classifier.decision_function(xy).reshape(X.shape) 
        plt.contour(X, Y, Z, colors = 'k', levels = [-1, 0, 1], alpha = 0.5) # decision boundary
        
if __name__ == "__main__" :
    Dataset=Problem_1().Load_data(["Case1_Class_0.txt", "Case1_Class_1.txt"]) # load data
    #Problem_1().Graph_Total_Train_Test(Dataset) # Plotting Train(70%), Test(30%), Total(100%) 
    Problem_1().SVM_support_vector_machine(Dataset, Result = 'Train', Kernel_Mode = 'rbf') 
    

    
    
    