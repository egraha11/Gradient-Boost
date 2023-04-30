import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
import sklearn.tree

class GradientBoost:

    def __init__(self):

        #number of individual trees to build 
        self.m = 100

        #maximum number of leaves per tree
        self.max_leaves = 15

        #list to hold all trees
        self.forest = []

        #list to hold all predictions, convert to numpy array when creating model
        self.preds = None

        #list to hold all residuals, convert to numpy array when creating model
        self.resids = None

        #learning rate
        self.learning_rate = .1

        #save intial prediction for future predictiosn once the models are complete
        self.initial_prediction = None


    def fit(self, x, y):

        self.initial_prediction = np.mean(y)

        #set initial value for all predictions 
        self.preds = np.full(len(x), self.initial_prediction)

        #set initial value for all residuals
        self.resids = y - self.preds

        for t in range(self.m):

            #create a tree that best predicts residuals 
            model = DecisionTreeRegressor(max_leaf_nodes=self.max_leaves)
            model_fit = model.fit(x, self.resids)

            #save new tree to forest
            self.forest.append(model_fit)


            #code to visualize trees included
            if (t == 0):

                fig = plt.figure(figsize=(25,20))
                _ = sklearn.tree.plot_tree(model_fit, 
                    feature_names=["v", "w", "x", "y", "z"],  
                    class_names="zz",
                    filled=True)

                fig.savefig("decistion_tree.png")




            #predict new value using old predictions and new leaf values scaled by the larning rate
            self.preds = np.add(self.preds, np.multiply(model_fit.predict(x), self.learning_rate))

            #calculate new residuals 
            self.resids = y - self.preds




    def predict(self,x):

        #create an empty dataframe to store predictions made by each tree
        df = pd.DataFrame(index=range(len(x)), columns=range(len(self.forest)))

        #fit each tree with features
        for tree in range(len(self.forest)):

            df.iloc[:, tree] = self.forest[tree].predict(x)

        #multiply each predictions by the learning rate 
        df= df * self.learning_rate

        #convert to numpy array
        array = np.array(df)

        #sum predictions made by each tree
        final_preds = self.initial_prediction + array.sum(axis=1)

        return final_preds  




    def accuracy(self, preds, actuals):

        #calculate mean absolute error
        mae = (sum((actuals - preds)))/len(preds)
        print("Mean Absolute Error: " + str(mae))

        #calculate mean square error
        mse = np.square(np.subtract(actuals,preds)).mean()
        print("Mean Square Error: " + str(mse))

        #calcualte mean absolute percentage error
        mape = round((sum((actuals - preds)/actuals))/len(preds), 2)
        print("Mean Absolute Percentage Error: " + str(mape) + "%")

        print("Actuals vs Preds: " + str(list(zip(actuals, preds))))



def main():

    #create a regressive dataset 
    x, y = make_regression(n_samples=150, n_features=5, noise=10)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25)

    model = GradientBoost()
    model.fit(x_train, y_train)

    final_preds = model.predict(x_test)

    model.accuracy(final_preds, y_test)





if __name__ == "__main__":
    main()