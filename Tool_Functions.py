# Tool_Functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score, KFold


def power_simulation(n, m, num_simulations=1000):

    np.random.seed(42)
    power = []
    esperances = []
    variances = []
    test_statistics = []
    test_alternative = []
    beta1 = 2
    beta2_values = np.linspace(0, 0.16, 9)
    delta_list = []

    for beta2 in beta2_values:

        differences = []

        for _ in range(num_simulations):
            epsilon = np.random.normal(0, 1, n + m)  # Generate epsilon inside the loop
            x = np.random.uniform(0, 5, n + m)
            y = beta1 * x + beta2 * x**2 + epsilon

            x_reshape = x.reshape(-1,1)

            x_train, x_test, y_train, y_test = train_test_split(x_reshape, y, test_size=m, random_state=None)

            # Linear Model
            model_a1 = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
            y_pred_a1 = model_a1.predict(x_test.reshape(-1, 1))
            mse_a1 = mean_squared_error(y_test, y_pred_a1)

            # Quadratic Model
            x_train_quad = np.column_stack((x_train, x_train**2))
            x_test_quad = np.column_stack((x_test, x_test**2))
            
            model_a2 = LinearRegression().fit(x_train_quad, y_train)
            y_pred_a2 = model_a2.predict(x_test_quad)
            mse_a2 = mean_squared_error(y_test, y_pred_a2)

            differences.append(mse_a1 - mse_a2)

        d_barre = sum(differences)/num_simulations 
        var = np.var(differences, ddof=1)
        test_stat = (num_simulations**0.5)*d_barre/np.sqrt(var) # Statistique pour l'hypothèse nulle

        delta = d_barre
        student = stats.t.ppf(0.95, df=num_simulations-1)
        mu = (num_simulations ** 0.5)*delta/np.sqrt(var)
        test_alt = student - mu
        puissance = 1 - stats.norm.cdf(student - mu, 0, 1)
       
        delta_list.append(delta)
        esperances.append(d_barre)
        variances.append(var)
        test_statistics.append(test_stat)
        test_alternative.append(test_alt)
        power.append(puissance)

    return esperances, variances, test_statistics, test_alternative, power


def power_competition(n,B,m=150):
        
   np.random.seed(42)
   esperances = []
   variances = []
   test_statistics = []
   test_alternative = []
   power = []
   delta_list = []
   epsilon = np.random.normal(0, 1, n)
   x = np.random.uniform(0, 5, n)
   beta1 = 2
   beta2_values = np.linspace(0, 0.16, 9)

   bootstrap_indices = np.random.choice(n, size=(B, n), replace=True) # bootstrapping

   for beta2 in beta2_values:
       
       y = beta1 * x + beta2 * x**2 + epsilon 

       x_sorted_idx = np.argsort(x)  # Indices pour trier x par ordre croissant
       x_sorted = x[x_sorted_idx]  # x trié
       y_sorted = y[x_sorted_idx]  # y correspondant trié

       # Génération de l'ensemble de test indépendant
       x_test = np.sort(np.random.choice(x_sorted, m, replace=True))  # Choix aléatoire avec remplacement
       y_test = np.interp(x_test, x_sorted, y_sorted)  # Interpolation pour obtenir y_test correspondant à x_test               
       
       db = []
       
       for bootstrap_index in bootstrap_indices:
       
          x_train = x[bootstrap_index]
          y_train = beta1 * x_train + beta2 * x_train**2 + epsilon[bootstrap_index]

          x_train_reshape = x_train.reshape(-1, 1) # fit pour le modèle 

          model_a1 = LinearRegression().fit(x_train_reshape, y_train) # Modèle linéaire
          y_pred_a1 = model_a1.predict(x_test.reshape(-1,1))
          p1b = mean_squared_error(y_test, y_pred_a1)

          x_train_quad = np.column_stack((x_train, x_train**2))
          x_test_quad = np.column_stack((x_test, x_test**2))

          model_a2 = LinearRegression().fit(x_train_quad, y_train)          
          y_pred_a2 = model_a2.predict(x_test_quad)
          p2b = mean_squared_error(y_test, y_pred_a2)

          db.append(p1b - p2b)  # Différence des moyennes de performance sur chaque bootstrap

       # Fin du boostrapping

       d_barre = sum(db)/B # Converge p.s vers l'espérance mu

       var = np.var(db, ddof=1)
       test_stat = (B**0.5)*d_barre/np.sqrt(var) # Statistique pour l'hypothèse nulle

       delta = np.mean(db)
       student = stats.t.ppf(0.95, df=B-1)
       mu = (B ** 0.5)*delta/np.sqrt(var)
       test_alt = student - mu
       puissance = 1 - stats.norm.cdf(student - mu, 0, 1)
       
       delta_list.append(delta)
       esperances.append(d_barre)
       variances.append(var)
       test_statistics.append(test_stat)
       test_alternative.append(test_alt)
       power.append(puissance)

   return esperances, variances, test_statistics, test_alternative, power

def power_RW_OOB(n,B):
        
   np.random.seed(42)
   esperances = []
   variances = []
   test_statistics = []
   test_alternative = []
   power = []
   epsilon = np.random.normal(0, 1, n)
   x = np.random.uniform(0, 5, n)
   beta1 = 2
   beta2_values = np.linspace(0, 0.16, 9)

   bootstrap_indices = np.random.choice(n, size=(B, n), replace=True) # bootstrapping

   for beta2 in beta2_values:
       
       y = beta1 * x + beta2 * x**2 + epsilon # generation de données
       db = []
       
       for bootstrap_index in bootstrap_indices:
       
          oob_indices = np.setdiff1d(np.arange(n), bootstrap_indices, assume_unique=True)
          x_train = x[bootstrap_index]
          y_train = beta1 * x_train + beta2 * x_train**2 + epsilon[bootstrap_index]

          mask = np.ones(len(x), dtype=bool)
          mask[bootstrap_index] = False

          x_test = x[mask]
          y_test = y[mask]

          x_train_reshape = x_train.reshape(-1, 1)
          x_test_reshape = x_test.reshape(-1,1)

          model_a1 = LinearRegression().fit(x_train_reshape, y_train) # Modèle linéaire
          y_pred_a1 = model_a1.predict(x_test_reshape.reshape(-1,1))
          p1b = mean_squared_error(y_test, y_pred_a1)

          x_train_quad = np.column_stack((x_train_reshape, x_train**2))
          x_test_quad = np.column_stack((x_test_reshape, x_test**2))


          model_a2 = LinearRegression().fit(x_train_quad, y_train)
          y_pred_a2 = model_a2.predict(x_test_quad)
          p2b = mean_squared_error(y_test, y_pred_a2)

          db.append(p1b - p2b)  # Différence des moyennes de performance sur chaque bootstrap

       # Fin du boostrapping

       d_barre = sum(db)/B # Converge p.s vers l'espérance mu

       var = np.var(db, ddof=1)
       test_stat = (B**0.5)*d_barre/np.sqrt(var) # Statistique pour l'hypothèse nulle

       delta = d_barre
       student = stats.t.ppf(0.95, df=B-1)
       mu = (B ** 0.5)*delta/np.sqrt(var)
       test_alt = student - mu
       puissance = 1 - stats.norm.cdf(student - mu, 0, 1)
       
       esperances.append(d_barre)
       variances.append(var)
       test_statistics.append(test_stat)
       test_alternative.append(test_alt)
       power.append(puissance)

   return esperances, variances, test_statistics, test_alternative, power


def power_RW_CV(n, B):
    np.random.seed(42)
    esperances = []
    variances = []
    test_statistics = []
    test_alternative = []
    power = []
    epsilon = np.random.normal(0, 1, n)
    x = np.random.uniform(0, 5, n)
    beta1 = 2
    beta2_values = np.linspace(0, 0.16, 9)
    
    # Generate bootstrap indices
    bootstrap_indices = np.random.choice(n, size=(B, n), replace=True)

    for beta2 in beta2_values:
        db = []

        for bootstrap_index in bootstrap_indices:
            # Generate data
            y = beta1 * x + beta2 * x**2 + epsilon

            x_train = x[bootstrap_index]
            y_train = beta1 * x_train + beta2 * x_train**2 + epsilon[bootstrap_index]
    
            # Cross-validation setup
            kf = KFold(n_splits=5)
            cv_scores_linear = []
            cv_scores_quadratic = []

            for train_index, test_index in kf.split(x_train):
                x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

                # Linear model
                model_a1 = LinearRegression().fit(x_train_cv.reshape(-1, 1), y_train_cv)
                y_pred_a1 = model_a1.predict(x_test_cv.reshape(-1, 1))
                mse_a1 = mean_squared_error(y_test_cv, y_pred_a1)
                cv_scores_linear.append(mse_a1)

                # Quadratic model
                x_train_cv_quad = np.column_stack((x_train_cv, x_train_cv**2))
                x_test_cv_quad = np.column_stack((x_test_cv, x_test_cv**2))
                model_a2 = LinearRegression().fit(x_train_cv_quad, y_train_cv)
                y_pred_a2 = model_a2.predict(x_test_cv_quad)
                mse_a2 = mean_squared_error(y_test_cv, y_pred_a2)
                cv_scores_quadratic.append(mse_a2)

            p1b = np.mean(cv_scores_linear)
            p2b = np.mean(cv_scores_quadratic)

            db.append(p1b - p2b)  # Difference in performance means for each bootstrap

        # End of bootstrapping
        d_barre = sum(db) / B  # Sample mean

        var = np.var(db, ddof=1)
        test_stat = (B**0.5) * d_barre / np.sqrt(var)  # Test statistic under null hypothesis

        delta = d_barre
        student = stats.t.ppf(0.95, df=B-1)
        mu = (B**0.5) * delta / np.sqrt(var)
        test_alt = student - mu
        puissance = 1 - stats.norm.cdf(student - mu, 0, 1)

        esperances.append(d_barre)
        variances.append(var)
        test_statistics.append(test_stat)
        test_alternative.append(test_alt)
        power.append(puissance)

    return esperances, variances, test_statistics, test_alternative, power

