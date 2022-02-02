# Statistical Analysis  
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

class Stat_Analysis:
    
    def __init__(self,DATA_Column): # Data_column
        self.DATA_Column=DATA_Column
    
    def Diabetes_Correlation_Heatmap (self, DATA): # Correlation_heatmap
        plt.figure(figsize = (10, 10))
        Correlation = DATA.corr() # correlation 
        sns.heatmap(Correlation, linewidth=0.5, annot = True, fmt= '.2f')
        plt.xticks(rotation = 0); plt.title("Correlation of Diabetes Factor", size=15)
        plt.show(); plt.tight_layout()
    
    def Diabetes_Boxplot (self, Diabetes_Patient, Non_Diabetes_Patient):
        Diamond_marker = dict(markerfacecolor='g', marker='D') # marker dictionary
        for Index in range(len(self.DATA_Column)):
            plt.figure(Index+2) # figure number 
            plt.boxplot([Non_Diabetes_Patient[self.DATA_Column[Index]], # box plot 
                          Diabetes_Patient[self.DATA_Column[Index]]], flierprops=Diamond_marker)
            plt.xlabel(self.DATA_Column[Index], fontsize = 12) 
            plt.xticks([1, 2], ['Diabetes', 'Non-Diabetes'], fontsize = 12); plt.grid()
        plt.show()
    
    def Diabetes_histogram_pdf(self, Diabetes_Patient, Non_Diabetes_Patient):
        plt.figure(figsize = (10, 10)) # fig_size 
        Non_Diabetes_Patient = sklearn.utils.shuffle(Non_Diabetes_Patient) # Data shuffle  
        for Index in range(len(self.DATA_Column)):
            plt.subplot(4, 2, Index+1) # auto subplot 
            sns.distplot(Diabetes_Patient[self.DATA_Column[Index]], bins=50, 
                         kde = True, kde_kws = {'color': 'red'}, hist_kws = {'color': 'coral'}) # Diabetes_Patient
            sns.distplot(Non_Diabetes_Patient[self.DATA_Column[Index]],bins=50, 
                         kde = True, kde_kws = {'color': 'blue'}, hist_kws = {'color': 'royalblue'}) # Non_Diabetes_Patient
            plt.xlabel(self.DATA_Column[Index], fontsize = 12) # Non_Diabetes vs Diabetes 
        plt.show();plt.tight_layout() 


