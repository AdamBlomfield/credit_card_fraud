# Dataframes
import pandas as pd
import numpy as np

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preparation
    # Train:Test
from sklearn.model_selection import train_test_split
    # Scaling
from sklearn.preprocessing import RobustScaler

# Resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Model Tuning and Cross Validation
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

# # Model metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.metrics import plot_confusion_matrix

# # Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def convert_array_to_dataframe(array, columns):
    '''Take a numpy array and convert it into a pandas dataframe'''
    # capture the input data type
    input_dtype = type(array)
    # Convert from array to dataframe
    array = pd.DataFrame(data= array, 
                 index= range(len(array)),   
                 columns= columns)
    # capture the output data type
    output_dtype = type(array)
    
    return array, input_dtype, output_dtype

def best_model_score(clf, param_grid, X_train, y_train, X_test, y_test, scoring='f1', cv=3, n_jobs=-1):
    '''Conducts Gridsearch to return the best test data accuracy and f1 score for a classifier as well as y_pred'''
    
    # Create GridSearch Object
    grid_clf = GridSearchCV(clf, 
                            param_grid, 
                            scoring='f1', 
                            cv=cv
                           )

    # Fit our GridSearch Object and pass  in the training data
    grid_clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = grid_clf.predict(X_test)
    y_pred_proba = grid_clf.predict_proba(X_test)
    
    # Best F1 Score
    best_f1 = f1_score(y_test, y_pred)
    print('Best Test Data F1 score: {}'.format(round(best_f1, 5)))
    
    # Best Recall Score
    best_recall = recall_score(y_test, y_pred)
    print('Best Test Data Recall score: {}'.format(round(best_recall, 5)))
    
    # Best Parameters
    if len(param_grid):
        print('\nOptimal parameters:')
        best_parameters = grid_clf.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print('\t{}: {}'.format(param_name, best_parameters[param_name]))    
    
#     # Classification report
#     print('\n', classification_report(y_test, y_pred))
    
    return best_f1, best_recall, y_pred, y_pred_proba
    

    
def combination_for_best_score(df):
    ''' Return the highest score and prints the associated combination of resampling and classifier'''
    index_positions = []
    max_score = df.max().max()
    
    # Find where the max F1 score is (NB could be more than one max value)
    df_bool_max_score = df.isin([max_score])
    # Find index/classifier that contains the max F1 score
    best_classifier_row = df_bool_max_score.any()
    best_classifier_name = list(best_classifier_row[best_classifier_row == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in best_classifier_name:
        rows = list(df_bool_max_score[col][df_bool_max_score[col] == True].index)
        for row in rows:
            index_positions.append((row, col))
    
    print('The highest score was {}'.format(round(max_score, 4)))
    best_classifier = index_positions[0][0]
    best_resampling = index_positions[0][1]
    print('This was achieved by resampling with {} and using the {} classifier'.format(best_resampling, best_classifier))
    
    return max_score


def resample_training_data(resample_method, X_train, y_train, title):
    '''Resamples the train data using the resample method specified'''
    
    # Resample training data
    resample_method.fit(X_train, y_train)
    X_train_resampled, y_train_resampled = resample_method.fit_resample(X_train, y_train)
    
    print('\nDistribution of Training Data {}:'.format(title))
    # Print new distribution of training data (absolute values)
    counter = Counter(y_train_resampled)   
    # Print new distribution of training data (ratio)
    ratio = round(counter[0]/counter[1], 1)
    print('\t0: {:<15}1: {:^10}{:>15} : 1'.format(counter[0], counter[1], ratio))
    
    return X_train_resampled, y_train_resampled

def custom_confusion_matrix(y_test, y_pred, model_name, resample_name, save_fig=False):
    '''Produces an elegant confusion matrix using seaborn heatmap'''
    cm = confusion_matrix(y_test, y_pred)
    f = plt.figure(figsize=(20,10))
    sns.heatmap(cm.T, square=True, annot=True, fmt = 'g', cmap='RdBu', cbar=False, xticklabels=['legitimate', 'fraud'], yticklabels=['legitimate', 'fraud'])
    plt.title('Confusion Matrix: {} Model\n(training data {})\n'.format(model_name, resample_name))
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig('cm_rf_smote')

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Confusion Matrix:",'\nTN:', tn, 'FP:', fp, 'FN:', fn, 'TP:', tp)
    
    # Save figure
    if save_fig:
        f.savefig('../visuals/confusion_matrix_{}_model_{}_resampling.pdf'.format(model_name, resample_name), bbox_inches='tight')    
    

def reshape_scores_df(df):
    '''Takes a df and configures it so its easier to do a barplot'''
    df = df.reset_index() # Move classifier names into 1st column
    df = df.iloc[1:,:] # Drop the baseline model
    classifiers = list(df.columns[-4:])
    df = df.melt(id_vars=['index'], value_vars=classifiers) # melt dataframe so its easier to plot
    return df

def scores_bar_plot(df, metric, figsize=(15,10)):
    '''takes a configured df and plots a barplot'''
    
    f = plt.figure(figsize=figsize);
    sns.set(font_scale=2, style='whitegrid');

    sns.barplot(x = 'variable', y = 'value', hue = 'index', data=df, ci=None);
    plt.ylabel('Score');
    plt.xlabel('\nResampling Method')
    plt.ylim(0,1)
    plt.title('Barplot of {} scores across different resampled datasets and classifiers\n'.format(metric))
    
    # Horizontal line to show max combination
    high_score = round(df.max().value, 4)
    plt.hlines(y=df['value'].max(), xmin=-0.4, xmax=3.4, ls='--', lw=3, label='Highest {} Score: {}'.format(metric, high_score));
    
    # Put the legend out of the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Save figure
    f.savefig('../visuals/barplot_of_{}_scores.pdf'.format(metric.lower()), bbox_inches='tight')
    
def plot_pr_curve(y_test, y_pred, y_pred_proba, title, save_fig=True):
    '''Plot the Precision-Recall Curve of a model'''
    
    # keep probabilities for the fraud outcome only
    y_pred_proba = y_pred_proba[:,1]
    
    # Calculate Precision and Recall values
    precision_scores, recall_scores, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_value = auc(recall_scores, precision_scores)
    f1_value = f1_score(y_test, y_pred)

    # summarize scores
    print('{}: F1: {}\tAUC: {}'.format(title, round(f1_value,4), round(auc_value,4)))

    
    f = plt.figure(figsize=(20,10))
    # Plot no skill line
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', lw=5)

    # plot the precision-recall curve
    plt.plot(recall_scores, precision_scores, marker='.', label='{}'.format(title), lw=5)
    
    # title & axis labels
    plt.title('Precision-Recall Curve for {}\nAUC: {}\n'.format(title, round(auc_value,4)))
    plt.xlabel('Recall')
    plt.xlim(0,1)
    plt.ylabel('Precision')
    plt.ylim(0,1)
    
    # show the legend
    plt.legend()
    
    # show the plot
    plt.show()
    
    # Save Plot
    if save_fig:
        f.savefig('../visuals/pr_curve_for_{}.pdf'.format(title), bbox_inches='tight')