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
import re
import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter
import csv
import sys
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from itertools import compress
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit

from bayes_opt import *



class ClassificationExperiments:
    def __init__(self,
                 scores_dir,
                 experiment = 'stratified',
                 input_modality = 'accelerometer',
                 method = 'PS1',
                 binary = True,
                 include_freq = False,
                 optimize = False,
                 load_params = None,
                 optimize_feature_space = False,
                 return_model = True,
                 random_state = 42):
     
        self.scores_dir = scores_dir
        self.experiment = experiment
        self.input_modality = input_modality
        self.method = method
        self.binary = binary
        self.include_freq = include_freq
        self.load_params = load_params
        self.optimize = optimize
        self.optimize_feature_space = optimize_feature_space
        self.return_model = return_model

        self.random_state = random_state

        # Placeholders for data and model
        self.scores_dataframe = None
        self.periodicity_scores, self.freq_feats, self.features, self.annotations, self.person_ids = None, None, None, None, None
        self.X_train, self.X_test, self.X_val = None, None, None
        self.y_train, self.y_test, self.y_val = None, None, None

        self.X_train_oversampled, self.y_train_oversampled = None, None
        self.model = None

        #Useful for visualizations
        self.test_predictions = None
        self.class_names = None
        self.best_params = None
        self.best_feature_selection = None

    def load_data(self):
        data_folder = self.scores_dir
        csv_files = sorted(data_folder.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSVs found in {data_folder}")
        
        dfs = []
        pat = re.compile(r'^(?:URI-)?(\d{3})')  # match '001...' or 'URI-001...'
        
        for f in csv_files:
            df = pd.read_csv(f)
            df["SourceFile"] = f.name
            stem = f.stem  
        
            m = pat.match(stem)
            if not m:
                raise ValueError(f"Could not parse PersonID from filename: {f.name}")
        
            df["PersonID"] = int(m.group(1))  # '001' -> 1, '006' -> 6
            dfs.append(df)
        
        big_df = pd.concat(dfs, ignore_index=True)
        
        # Coerce numerics
        if self.input_modality == 'accelerometer':
            num_cols = ["Annotation"] + [f"Torso_{i}" for i in range(1, 11)] + \
                [f"LWrist_{i}" for i in range(1, 11)] + \
                [f"RWrist_{i}" for i in range(1, 11)] + \
                ["Torso_Period", "LWrist_Period", "RWrist_Period"]
            
        elif self.input_modality == 'pose':
            num_cols = ["Annotation"] +  [f"Head_{i}" for i in range(1, 11)] + \
                [f"RWrist_{i}" for i in range(1, 11)] + \
                [f"LWrist_{i}" for i in range(1, 11)] + \
                [f"RShoulder_{i}" for i in range(1, 11)] + \
                [f"LShoulder_{i}" for i in range(1, 11)] + \
                [f"Chest_{i}" for i in range(1, 11)] + \
                [f"Head_Accel_{i}" for i in range(1, 11)] + \
                [f"RWrist_Accel_{i}" for i in range(1, 11)] + \
                [f"LWrist_Accel_{i}" for i in range(1, 11)] + \
                [f"RShoulder_Accel_{i}" for i in range(1, 11)] + \
                [f"LShoulder_Accel_{i}" for i in range(1, 11)] + \
                [f"Chest_Accel_{i}" for i in range(1, 11)]
        else: 
            raise ValueError(f"Not a valid input modality")
            
        for c in num_cols:
            if c in big_df.columns:
                big_df[c] = pd.to_numeric(big_df[c], errors="coerce")
        
        # Ensure Session is string (or derive from filename if missing)
        if "Session" in big_df.columns:
            big_df["Session"] = big_df["Session"].astype(str)
        else:
            big_df["Session"] = big_df["SourceFile"].str.replace(r"\.csv$", "", regex=True)
            
        self.scores_dataframe = big_df

        if self.input_modality == 'accelerometer':
            tda_cols = [f"Torso_{i}" for i in range(1, 11)] + \
                [f"LWrist_{i}" for i in range(1, 11)] + \
                [f"RWrist_{i}" for i in range(1, 11)] 
            if self.include_freq == True:
                period_cols = ["Torso_Period", "LWrist_Period", "RWrist_Period"]
                self.freq_feats = big_df[period_cols].to_numpy()
            
        elif self.input_modality == 'pose':
            tda_cols = [f"Head_{i}" for i in range(1, 11)] + \
                [f"RWrist_{i}" for i in range(1, 11)] + \
                [f"LWrist_{i}" for i in range(1, 11)] + \
                [f"RShoulder_{i}" for i in range(1, 11)] + \
                [f"LShoulder_{i}" for i in range(1, 11)] + \
                [f"Chest_{i}" for i in range(1, 11)] + \
                [f"Head_Accel_{i}" for i in range(1, 11)] + \
                [f"RWrist_Accel_{i}" for i in range(1, 11)] + \
                [f"LWrist_Accel_{i}" for i in range(1, 11)] + \
                [f"RShoulder_Accel_{i}" for i in range(1, 11)] + \
                [f"LShoulder_Accel_{i}" for i in range(1, 11)] + \
                [f"Chest_Accel_{i}" for i in range(1, 11)]
        else: 
            raise ValueError(f"Not a valid input modality")
            
        X = big_df[tda_cols].to_numpy()

        if self.method == 'PS1':
            num_total_features = len(X[0])
            num_feature_groups_10 = int(num_total_features/10)
            group_sizes = [10] * num_feature_groups_10
            X_compressed = self.compress_features(X, group_sizes)
            self.periodicity_scores = X_compressed
            
        elif self.method == 'PS10':
            self.periodicity_scores = X

        else:
            raise ValueError(f"Not a valid periodicity score method")

        if self.input_modality == 'accelerometer':
            if self.include_freq == True:
                self.features = np.column_stack((self.periodicity_scores, self.freq_feats))
            else:
                self.features = self.periodicity_scores
        elif self.input_modality == 'pose':
            self.features = self.periodicity_scores
        else:
            raise ValueError(f"Not a valid input modality")

        self.person_ids = self.scores_dataframe["PersonID"].to_numpy()
        self.annotations = self.scores_dataframe['Annotation'].to_numpy()
        
    def stratified_split(self):

        person_list = [1, 2, 3, 4, 5, 6]
        
        indices_dict = {}
        
        for person in person_list:
            indices_dict[person] = np.where(self.person_ids == person)[0]
            
        X_train = []
        X_test = []
        X_val = []
        y_train = []
        y_test = []
        y_val = []
        
        for person, indices in indices_dict.items():
            person_annos = self.annotations[indices]
            person_features = self.features[indices]
            valid_classes = [cls for cls, count in Counter(person_annos).items() if count > 1]
            mask = [label in valid_classes for label in person_annos]
            person_annos = person_annos[mask]
            person_features = person_features[mask]
            X_temp, x_test , y_temp, Y_test = train_test_split(person_features, person_annos, test_size=0.2, random_state=42, stratify = person_annos)
            x_train, x_val, Y_train, Y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state = 42, stratify = y_temp)
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
        train_mask = y_train != -1
        test_mask = y_test != -1
        val_mask = y_val != -1

        self.X_train, self.X_test, self.X_val = X_train[train_mask], X_test[test_mask], X_val[val_mask]
        self.y_train, self.y_test, self.y_val = y_train[train_mask], y_test[test_mask], y_val[val_mask]

    def oversampler(self):
        '''
        Oversmaples minority classes to improve class imbalance problem for model training
    
        :Param y_train: Training annotations
    
        :Returns X_train_resampled, y_train_resampled: Oversampled training data
        '''
        
        counts = pd.Series(self.y_train).value_counts()
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
        if self.binary:
            random_oversampler = RandomOverSampler(sampling_strategy=sampling_strategy,random_state=42)
        else:
            #This will upsample the minority classes to be the same cardinatlity of the majority (25/25/25/25)
            random_oversampler = RandomOverSampler(random_state=42)
    
        self.X_train_oversampled, self.y_train_oversampled = random_oversampler.fit_resample(self.X_train, self.y_train)


    def compress_features(self, X, group_sizes):
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

    def stratified_set_classification(self):

        if self.binary:
            self.y_train_oversampled = (self.y_train_oversampled != 0).astype(int)
            self.y_test = (self.y_test != 0).astype(int)
            self.y_val = (self.y_val != 0).astype(int)

        if self.optimize:
            bayes_optimizer = BayesianOptimizer(n_calls = 100, n_random_starts = 25, method = self.method, feature_selection = self.optimize_feature_space)
            bayes_optimizer.do_bayes_opt(self.X_train_oversampled, self.y_train_oversampled, self.X_val, self.y_val)
            params = bayes_optimizer.best_params
            if self.optimize_feature_space:
                self.best_feature_selection = bayes_optimizer.best_feature_mask
                self.best_params = bayes_optimizer.best_params
                self.X_train_oversampled = bayes_optimizer.select_features(self.X_train_oversampled, self.best_feature_selection)
                self.X_test = bayes_optimizer.select_features(self.X_test, self.best_feature_selection)
            else:
                self.best_params = bayes_optimizer.best_params
        else:
            self.best_params = self.load_params

        model = RandomForestClassifier(**self.best_params, n_jobs=-1, random_state=42)

        model.fit(self.X_train_oversampled, self.y_train_oversampled)

        test_predictions = model.predict(self.X_test)

        if self.binary:
            class_names = ['None', 'SMM']
        else:
            class_names = ['None', 'Rock', 'Flap', 'Flap-Rock']

        self.test_predictions = test_predictions
        self.class_names = class_names
        print(classification_report(self.y_test, test_predictions, target_names=class_names))
        self.model = model

    def leave_one_out(self):
        
        if self.experiment == 'LOCO':
            groups = self.scores_dataframe["PersonID"].astype(int).to_numpy()
        elif self.experiment == 'LOSO':
            s_session = self.scores_dataframe.get("Session", pd.Series("", index=self.scores_dataframe.index)).astype(str)
            s_source  = self.scores_dataframe.get("SourceFile", pd.Series("", index=self.scores_dataframe.index)).astype(str)
            is_uri = s_session.str.startswith("URI-") | s_source.str.startswith("URI-")
            study = np.where(is_uri, "URI", "NonURI")
    
            group_key = pd.Series(study, index=self.scores_dataframe.index).astype(str) + "-" + self.scores_dataframe["PersonID"].astype(str)
    
            groups, _ = pd.factorize(group_key)
        else:
            raise ValueError(f"Not a valid experiment")
            
        return groups

        
    def leave_one_out_classification(self):

        groups = self.leave_one_out()

        logo = LeaveOneGroupOut()

        for train_val_idx, test_idx in logo.split(self.features, self.annotations, groups=groups):
            print(f"\n--- NEW FOLD ---")
            test_group = np.unique(groups[test_idx])
            print(f"Leaving out: {test_group}")
            X_test = self.features[test_idx]
            y_test = self.annotations[test_idx]
        
            X_train_val = self.features[train_val_idx]
            y_train_val = self.annotations[train_val_idx]

            person_train_val = self.person_ids[train_val_idx]
            ann_train_val = self.annotations[train_val_idx]

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

            self.X_train, self.X_test, self.X_val = X_train[train_mask], X_test[test_mask], X_val[val_mask]
            self.y_train, self.y_test, self.y_val = y_train[train_mask], y_test[test_mask], y_val[val_mask]

            self.oversampler()

            if self.binary:
                self.y_train_oversampled = (self.y_train_oversampled != 0).astype(int)
                self.y_test = (self.y_test != 0).astype(int)
                self.y_val = (self.y_val != 0).astype(int)
        
            if self.optimize:
                bayes_optimizer = BayesianOptimizer(n_calls = 100, n_random_starts = 25, method = self.method, feature_selection = self.optimize_feature_space)
                bayes_optimizer.do_bayes_opt(self.X_train_oversampled, self.y_train_oversampled, self.X_val, self.y_val)
                if self.optimize_feature_space:
                    self.best_feature_selection = bayes_optimizer.best_feature_mask
                    self.best_params = bayes_optimizer.best_params
                    self.X_train_oversampled = bayes_optimizer.select_features(self.X_train_oversampled, self.best_feature_selection)
                    self.X_test = bayes_optimizer.select_features(self.X_test, self.best_feature_selection)
                else:
                    self.best_params = bayes_optimizer.best_params
            else:
                self.best_params = self.load_params

            model = RandomForestClassifier(**self.best_params, n_jobs=-1, random_state=42)

            model.fit(self.X_train_oversampled, self.y_train_oversampled)
            
            # Predict on test set
            test_predictions = model.predict(self.X_test)
            
            # Compute accuracy
            accuracy_test = accuracy_score(self.y_test, test_predictions)
            print(f"Group: {test_group} Results")
            if self.binary:
                class_names = ['None', 'SMM']
            else:
                class_names = ['None', 'Rock', 'Flap', 'Flap-Rock']

            print(classification_report(self.y_test, test_predictions, target_names=class_names))

            
    def run_classification_experiment(self):
        self.load_data()
        if self.experiment == 'stratified':
            self.stratified_split()
            self.oversampler()
            self.stratified_set_classification()
        elif self.experiment == 'LOSO' or self.experiment == 'LOCO':
            self.leave_one_out_classification()
        else:
            raise ValueError(f"Not a valid experiment")






