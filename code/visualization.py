import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

# analysis
class StatAnalysis:
    def __init__(self, data_columns):
        self.data_columns = data_columns

    # correlation heatmap visualization
    def plot_correlation_heatmap(self, data):
        plt.figure(figsize=(10, 10))
        correlation = data.corr()
        sns.heatmap(correlation, linewidth=0.5, annot=True, fmt='.2f')
        plt.xticks(rotation=0)
        plt.title("Correlation of Diabetes Factors", size=15)
        plt.tight_layout()
        plt.show()

    # boxplot visualization
    def plot_boxplots(self, diabetes_patients, non_diabetes_patients):
        marker_style = dict(markerfacecolor='g', marker='D')
        for index, column in enumerate(self.data_columns, start=2):
            plt.figure(index)
            plt.boxplot([non_diabetes_patients[column], diabetes_patients[column]], 
                        flierprops=marker_style)
            plt.xlabel(column, fontsize=12)
            plt.xticks([1, 2], ['Non-Diabetes', 'Diabetes'], fontsize=12)
            plt.grid()
        plt.show()

    # histogram visualization
    def plot_histograms_and_pdfs(self, diabetes_patients, non_diabetes_patients):
        plt.figure(figsize=(10, 10))
        shuffled_non_diabetes = shuffle(non_diabetes_patients)
        
        for index, column in enumerate(self.data_columns, start=1):
            plt.subplot(4, 2, index)
            sns.distplot(diabetes_patients[column], bins=50, 
                         kde=True, kde_kws={'color': 'red'}, hist_kws={'color': 'coral'})
            sns.distplot(shuffled_non_diabetes[column], bins=50, 
                         kde=True, kde_kws={'color': 'blue'}, hist_kws={'color': 'royalblue'})
            plt.xlabel(column, fontsize=12)
        
        plt.tight_layout()
        plt.show()