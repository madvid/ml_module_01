import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

import os
import sys

path1 = os.path.join(os.path.dirname(__file__), '..', 'ex03')
path2 = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path1)
sys.path.insert(1, path2)

from my_linear_regression import MyLinearRegression as MyLR
from plot import plot
from prediction import predict_

if __name__ == '__main__':
    #############################################################
    # ____________________  FIRST PART  _______________________ #
    #############################################################
    # Read the CSV data file:
    try:
        data = pd.read_csv('are_blue_pills_magics.csv')
    except:
        print("An error occured during the reading of the dataset.")
        sys.exit()
    
    # Checking the dataset:
    cols = data.columns
    if not all([c in ["Patient", "Micrograms", "Score"] for c in cols]):
        print("Unexpected column in the dataset.")
        sys.exit()
    
    try:
        x = data.Micrograms.values.reshape(-1,1)
        y = data.Score.values.reshape(-1,1)
        thetas = np.random.rand(2,1)
        mylr = MyLR(thetas, alpha=5e-2, max_iter=1000)
        mylr.fit_(x, y)
        print("valeur de thetas:", thetas)
        print("valeur de self.thetas:", mylr.thetas)
        plot(x, y, mylr.thetas, b_legend = True,
             axes_labels=["Quantity of blue pill (in micrograms)", "Space driving score"],
             data_labels={"raw":r"$S_{true}$(pills)", "prediction":r"$S_{predict}$(pills)"})
    except:
        print("Something wrong happened during model instance or training.")
        sys.exit()

    #############################################################
    # ___________________  SECOND PART  _______________________ #
    #############################################################
    # Loss function vizualisation:
    n = 6
    theta0 = np.linspace(80, 96, n)
    theta1 = np.linspace(-14, -4, 100)
    
    viridis = get_cmap('viridis', n)
    fig, axe = plt.subplots(1,1, figsize = (15,10))
    for t0, color in zip(theta0, viridis(range(n))):
        l_loss = []
        for t1 in theta1:
            ypred = predict_(x, np.array([[t0], [t1]]))
            l_loss.append(MyLR.loss_(y, ypred))
        axe.plot(theta1, np.array(l_loss), label = r"J($\theta_0$ = " + f"{t0}," + r"$\theta_1$)", lw = 2.5, c=color)
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"cost function J($\theta_0 , \theta_1$)")
    axe.set_ylim([10, 150])
    plt.show()

    #############################################################
    # ____________________  THIRD PART  _______________________ #
    #############################################################
    # MSE Calculation
    ## Based on the graph in part 2, we choose for h: t0 = 89.6 and t1 = -9:
    graphical_choice_theta = np.array([[89.6], [-9]])
    graph_mse = MyLR.mse_(predict_(x,graphical_choice_theta), y)
    print("value of mse with thetas choosen based on the graph: ", graph_mse)

    # MSE of the trained model:
    trained_mse = MyLR.mse_(predict_(x, mylr.thetas), y)
    print("value of mse with thetas choosen based on the graph: ", trained_mse)