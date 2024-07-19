import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# svm class
class SVMProblem:
    def __init__(self, data_path='PATH'):
        self.data_path = data_path

    # data load
    def load_data(self, file_list):
        x_data, y_data = [], []
        for file_name in file_list:
            with open(self.data_path + file_name) as file:
                lines = file.readlines()
                x_data.extend([list(map(float, line.split())) for line in lines])
                y_data.extend([int(file_name[-5])] * len(lines))
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return train_test_split(x_data, y_data, test_size=0.3, random_state=1)

    # data visualization # EDA
    def plot_data(self, datasets, titles):
        plt.figure(figsize=(12, 4))
        for i, (x, y, title) in enumerate(zip(datasets[::2], datasets[1::2], titles), 1):
            plt.subplot(1, 3, i)
            plt.title(title)
            colors = ['blue' if label == 0 else 'red' for label in y]
            plt.scatter(x[:, 0], x[:, 1], c=colors)
        plt.tight_layout()
        plt.show()

    # training based on grid search
    def train_svm(self, x_train, y_train, kernel='rbf'):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        clf = GridSearchCV(SVC(kernel=kernel), param_grid, refit=True, return_train_score=True)
        clf.fit(x_train, y_train)
        return clf

    # result
    def print_results(self, clf):
        scores_df = pd.DataFrame(clf.cv_results_)
        print(scores_df[['param_C', 'param_gamma', 'mean_train_score', 'mean_test_score', 'rank_test_score']])
        print(f"Optimized parameters: {clf.best_params_}\nBest mean test score: {clf.best_score_}")

    # boundry plot
    def plot_decision_boundary(self, clf, x, y):
        plt.figure(figsize=(8, 8))
        colors = ['blue' if label == 0 else 'red' for label in y]
        plt.scatter(x[:, 0], x[:, 1], c=colors)

        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5)
        plt.show()

if __name__ == "__main__":
    problem = SVMProblem()
    x_train, x_test, y_train, y_test = problem.load_data(["Case1_Class_0.txt", "Case1_Class_1.txt"])
    
    clf = problem.train_svm(x_train, y_train)
    problem.print_results(clf)
    problem.plot_decision_boundary(clf, x_train, y_train)