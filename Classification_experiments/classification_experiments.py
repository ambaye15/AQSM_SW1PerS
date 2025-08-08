# Copyright 2025 Austin Amadou MBaye
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import pandas as pd
import numpy as np
from collections import Counter
import csv
import sys
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import compress
from joblib import dump, load
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def plot_PR_curve(best_model, X_test, y_test, binary=False):
    y_score = best_model.predict_proba(X_test)

    if binary:
        class_names = ['Normal', 'SMM']

        # Use probability of the positive class
        y_score_pos = y_score[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_score_pos)
        avg_precision = average_precision_score(y_test, y_score_pos)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, color='darkorange', label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision–Recall Curve (Normal vs. SMM)')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.tight_layout()
        plt.show()

    else:
        class_names = ['None', 'Rock', 'Flap', 'Flap-Rock']
        classes = np.array([0, 1, 2, 3])
        y_test_bin = label_binarize(y_test, classes=classes)

        precision = dict()
        recall = dict()
        avg_precision = dict()

        for i in range(len(classes)):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

        plt.figure(figsize=(10, 8))
        for i in range(len(classes)):
            plt.plot(recall[i], precision[i], lw=2, label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Multiclass Precision–Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(y_test, test_predictions, class_names, binary_method = False):
    
    accuracy_test = accuracy_score(y_test, test_predictions)

    cm = confusion_matrix(y_test, test_predictions)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    
    if binary_method:
        plt.title(f"Pose Accuracy Binary: {accuracy_test:.3f}")
    else:
        plt.title(f"Pose Accuracy Multiclass: {accuracy_test:.3f}")
        
    plt.tight_layout()
    plt.show()
    

def train_class_oversampling(X_train, y_train, binary_method = False):
    '''
    Oversmaples minority classes to improve class imbalance problem for model training

    :Param y_train: Training annotations

    :Returns X_train_resampled, y_train_resampled: Oversampled training data
    '''

    num_total_features = len(X_train[0])

    num_feature_groups_10 = int(len(X_train[0])/10)  # 20 groups of 10 features each

    group_sizes = [10] * num_feature_groups_10
    num_feature_groups = len(group_sizes)  # Total number of groups
    
    counts = pd.Series(y_train).value_counts()
    majority_class = counts.idxmax()
    majority_count = counts[majority_class]
    
    # Compute new total size based on majority class being 50%
    desired_total = majority_class_count = counts[majority_class] * 2
    
    target_counts = {majority_class: counts[majority_class]}
    
    # Each minority class will be about 17% of total distribution
    minority_classes = [cls for cls in counts.index if cls != majority_class]
    
    minority_target_count = int(0.17 * (majority_count := counts[majority_class]) / 0.50)
    
    for cls in minority_classes:
        target = max(counts[cls], minority_target_count)  # only oversample if needed
        minority_target_count = target  # Ensures consistency for all minority classes
        target = target if target > counts[cls] else counts[cls]
        counts[cls] = target
        target = minority_target_count
    
    sampling_strategy = {
        majority_class: majority_count,
        **{cls: minority_target_count for cls in minority_classes}
    }

    if binary_method:
        oversampler = RandomOverSampler(sampling_strategy=sampling_strategy,random_state=42)
    else:
        #This will upsample the minority classes to be the same cardinatlity of the majority (25/25/25/25)
        oversampler = RandomOverSampler(random_state=42)

    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled


def construct_feature_mask(params):
    '''
    Function to construct the feature mask which is used in optimization
    '''
    return [1 if i in required_feature_groups else params.get(f"feature_{i}", 0)
            for i in range(num_feature_groups)]


def select_features(X, feature_mask, group_sizes):
    '''
    Extract selected features based on binary mask used in optimization
    '''
    selected_indices = []
    start_idx = 0
    
    for i, include in enumerate(feature_mask):
        group_size = group_sizes[i]  
        if include == 1 or i in required_feature_groups:  # Always include required groups
            selected_indices.extend(range(start_idx, start_idx + group_size))
        start_idx += group_size  
        
    return X[:, selected_indices] if selected_indices else X  
    

def compress_features(X, group_sizes):
    '''
    Used to compute the PS1 scores for each sensor from the original vector of 10 scores per sensor
    '''
    compressed = []
    start_idx = 0
    for group_size in group_sizes:
        group = X[:, start_idx:start_idx + group_size]
        # Subtract column 1 from column 0 in this group
        diff = group[:, 0] - group[:, 1]
        compressed.append(diff.reshape(-1, 1)) 
        start_idx += group_size
    return np.hstack(compressed)
    

# Section: Stratified Random Sampling Model Training 
# ----------------------------------------------------


def test_train_val_split(df, num_columns = 120):
    '''
    Splits the data into testing, training and validation ensuring even sampling from each individual and each class

    :Param df: dataframe that contains TDA-derived scores

    :Returns test/train/val split
    '''
    tda_cols = [f"TDA_{i+1}" for i in range(num_columns)]
    
    tda_features = df[tda_cols].to_numpy()
    
    person_ids = df["Person_ID"].to_numpy()
    annotations = df['Annotation_1'].to_numpy()
    
    person_list = [1, 2, 3, 4, 5, 6]

    indices_dict = {}
    
    for person in person_list:
        indices_dict[person] = np.where(person_ids == person)[0]
        
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []
    
    for person, indices in indices_dict.items():
        person_annos = annotations[indices]
        person_features = tda_features[indices]
        valid_classes = [cls for cls, count in Counter(person_annos).items() if count > 1]
        mask = [label in valid_classes for label in person_annos]
        person_annos = person_annos[mask]
        person_features = person_features[mask]
        X_temp, x_test , y_temp, Y_test = train_test_split(person_features, person_annos, test_size=0.2, random_state=42, stratify = person_annos)
        x_train, x_val, Y_train, Y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify = y_temp)
        X_train.extend(x_train)
        X_test.extend(x_test)
        X_val.extend(x_val)
        y_train.extend(Y_train)
        y_test.extend(Y_test)
        y_val.extend(Y_val)
        
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    X_val = np.vstack(X_val)
    y_train = np.vstack(y_train).ravel()
    y_test = np.vstack(y_test).ravel()
    y_val = np.vstack(y_val).ravel()

    return X_train, X_val, X_test, y_train, y_val, y_test


def do_bayes_stratified(data_file, PS1 = True, binary_method = False, doBayes = True, plotCM = False, plotPR = False, saveModel = False):
    '''
    Perform Bayesian Optimization to find best hyperparameters and features needed to train Random Forest model.
    '''
    
    if binary_method:
        class_names = ['None', 'SMM']
    else:
        class_names = ['None', 'Rock', 'Flap', 'Flap-Rock']

    df = pd.read_csv(data_file) 
    
    X_train, X_val, X_test, y_train, y_val, y_test = test_train_val_split(df)

    # Make absolute sure the only labels are {0,1,2,3}
    train_mask = y_train != -1
    test_mask = y_test != -1
    val_mask = y_val != -1
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]

    num_total_features = len(X_train[0])
    
    num_feature_groups_10 = int(num_total_features/10)
    
    group_sizes = [10] * num_feature_groups_10

    # This is for using PS1 scores rather than PS10
    if PS1:
        X_train = compress_features(X_train, group_sizes)
        X_val = compress_features(X_train, group_sizes)
        X_test = compress_features(X_test, group_sizes)


    X_train_resampled, y_train_resampled = train_class_oversampling(X_train, y_train, binary_method = binary_method)

    #If binary classification
    if binary_method:
        y_train_resampled = (y_train_resampled != 0).astype(int)
        y_test = (y_test != 0).astype(int)
        y_val = (y_val != 0).astype(int)

     # Define hyperparameter search space
    hyperparameter_space = [
        Integer(1, 200, name="n_estimators"),
        Integer(10, 200, name="max_depth"),
        Real(0.0001, 0.5, name="min_samples_split"),  
        Real(0.0001, 0.5, name="min_samples_leaf"),  
        Categorical(["sqrt", "log2", None], name="max_features"),
        Categorical(["balanced", "balanced_subsample"], name="class_weight"),
        Categorical(["gini", "entropy", "log_loss"], name="criterion"),]

    #To make optimization run faster, it is best to pre-select features you think will contribute the most to model classification
    required_feature_groups = {0, 6, 7, 8}

    # Optimize only for features not in required_feature_groups
    feature_selection_space = [
        Integer(0, 1, name=f"feature_{i}") for i in range(num_feature_groups) if i not in required_feature_groups
    ]

    #Combine both feature selection and hyperparameter tuning into the search space. NOTE: we found that most of the time the model will keep all sensors so one could only tune hyperparameters which would save time
    search_space = feature_selection_space + hyperparameter_space

    @use_named_args(search_space)
    def objective(**params):
        
        feature_mask = construct_feature_mask(params)
        
        hyperparams = {k: v for k, v in params.items() if k not in [f"feature_{i}" for i in range(num_feature_groups)]}
    
        X_train_selected = select_features(X_train_resampled, feature_mask)
        X_val_selected = select_features(X_val, feature_mask)
            
        model = RandomForestClassifier(**hyperparams, random_state=42, n_jobs=-1)
        
        model.fit(X_train_selected,y_train_resampled)
    
        # Predict on the validation set
        y_val_pred = model.predict(X_val_selected)
    
        val_f1 = f1_score(y_val, y_val_pred, average="macro")
        
        return -val_f1 
    
    # Perform Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=500, #Total number of iterations
        n_random_starts=150,   # Random samples before using Bayesian optimization to build probabilistic model
        random_state=42)

    #Get all the best parametersevaluated by optimization
    best_params = dict(zip([dim.name for dim in search_space], result.x))

    best_feature_mask = construct_feature_mask(best_params)
    
    best_hyperparams = {k: v for k, v in best_params.items() if not k.startswith("feature_")}
    
    print("\nBest Feature Groups Selected:", list(compress(range(num_feature_groups), best_feature_mask)))
    print("\nBest Hyperparameters:", best_hyperparams)

    #Do final model evalutation on testing data:
    X_train_final = select_features(X_train_resampled, best_feature_mask)

    X_test_final = select_features(X_test, best_feature_mask)    

    best_model = RandomForestClassifier(**best_hyperparams, n_jobs=-1, random_state=42)

    best_model.fit(X_train_final, y_train_resampled)
    
    test_predictions = best_model.predict(X_test_final)
    
    accuracy_test = accuracy_score(y_test, test_predictions)
    
    # Print classification report
    print(classification_report(y_test, test_predictions, target_names=class_names))

    if plotCM:
        plot_confusion_matrix(y_test, test_predictions, class_names, binary_method = binary_method)

    if plotPR:
        plot_PR_curve(best_model, X_test_final, y_test, binary=binary_method)

    if saveModel:
        import joblib
        # Save trained model
        joblib.dump(best_model, "optimized_model.pkl")
        print(f"Model Saved")


# Section: Leave-One-Out Model Training 
# ----------------------------------------------------

from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit


def loso_groups(df, LOSO_method = 'Child'):
    '''
    Helper function to determine which version of LOSO we plan on doing
    '''
    if LOSO_method == 'Child':
        groups = df["Person_ID"].astype(int).to_numpy()
    else:
        groups = df["Child_Study_ID"].to_numpy()
    return groups


def leave_one_out_bayes(data_file, LOSO_method = 'Child', PS1 = True, binary_method = False, plotCM = False, plotPR = False):
    '''
    Perform Bayesian Optimization to find best hyperparameters and features needed to train Random Forest model while doing Leave-One-Out.
    
    - Param LOSO_method: (Default = 'Child') This selects whether to do LOSO of children. If LOSO_method = 'Session' then will do LOSO based on session like in other papers.
    '''

    if binary_method:
        class_names = ['None', 'SMM']
    else:
        class_names = ['None', 'Rock', 'Flap', 'Flap-Rock']
        
    df = pd.read_csv(data_file) 

    tda_cols = [f"TDA_{i+1}" for i in range(120)]

    X = df[tda_cols].to_numpy()

    num_total_features = len(X[0])
    
    num_feature_groups_10 = int(num_total_features/10)
    
    group_sizes = [10] * num_feature_groups_10

    if PS1:
        X = compress_features(X, group_sizes)
    
    y = df['Annotation_1'].to_numpy()
    
    person_ids = df["Person_ID"].to_numpy()

    groups = loso_groups(df, LOSO_method = LOSO_method)
    
    logo = LeaveOneGroupOut()
    
    for train_val_idx, test_idx in logo.split(X, y, groups=groups):
        print(f"\n--- NEW FOLD ---")
        test_group = np.unique(groups[test_idx])
        print(f"Leaving out: {test_group}")
        
        # Split into test and train/val
        X_test = X[test_idx]
        y_test = y[test_idx]
    
        X_train_val = X[train_val_idx]
        y_train_val = y[train_val_idx]
        person_train_val = person_ids[train_val_idx]
        ann_train_val = y[train_val_idx]
    
        # Composite stratification label: ensures balanced (person_id, annotation) pairs
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(X_train_val, y_train_val))
    
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]

        # Now filter out -1 labels
        train_mask = y_train != -1
        test_mask = y_test != -1
        val_mask = y_val != -1
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]
    
    
        X_train_resampled, y_train_resampled = train_class_oversampling(X_train, y_train, binary_method = binary_method)
    
        #If binary classification
        if binary_method:
            y_train_resampled = (y_train_resampled != 0).astype(int)
            y_test = (y_test != 0).astype(int)
            y_val = (y_val != 0).astype(int)

        hyperparameter_space = [
            Integer(1, 200, name="n_estimators"),
            Integer(10, 200, name="max_depth"),
            Real(0.0001, 0.5, name="min_samples_split"),  
            Real(0.0001, 0.5, name="min_samples_leaf"),  
            Categorical(["sqrt", "log2", None], name="max_features"),
            Categorical(["balanced", "balanced_subsample"], name="class_weight"),
            Categorical(["gini", "entropy", "log_loss"], name="criterion"),]
        
        @use_named_args(hyperparameter_space)
        def objective(**params):
                
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            
            model.fit(X_train_resampled,y_train_resampled)
        
            # Predict on the validation set
            y_val_pred = model.predict(X_val)
        
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            
            return -val_f1 
        
        # Perform Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=hyperparameter_space,
            n_calls=50, 
            n_random_starts=15,  
            random_state=42)

        best_params = dict(zip([dim.name for dim in hyperparameter_space], result.x))
        
        print("\nBest Hyperparameters:", best_params)
        
        # Train the best model with the optimal parameters
        best_model = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)
    
        best_model.fit(X_train_resampled, y_train_resampled)
        
        # Predict on test set
        test_predictions = best_model.predict(X_test)
        
        # Compute accuracy
        accuracy_test = accuracy_score(y_test, test_predictions)
        print(f"Group: {test_group} Results")

        # Print classification report
        print(classification_report(y_test, test_predictions, target_names=class_names))

        if plotCM:
            plot_confusion_matrix(y_test, test_predictions, class_names, binary_method = binary_method)
            
        if plotPR:
            plot_PR_curve(best_model, X_test, y_test, binary_method)

