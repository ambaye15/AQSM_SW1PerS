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

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


class BayesianOptimizer:
    def __init__(self, 
                 n_calls = 200,
                 n_random_starts = 30,
                 method = 'PS1',
                 feature_selection = False,
                 required_features = {0, 6, 7, 8},
                 random_state = 42):

        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.method = method
        self.feature_selection = feature_selection
        self.required_features = required_features

        self.hyperparameter_space = [
                Integer(1, 200, name="n_estimators"),
                Integer(10, 200, name="max_depth"),
                Real(0.0001, 0.5, name="min_samples_split"),  
                Real(0.0001, 0.5, name="min_samples_leaf"),  
                Categorical(["sqrt", "log2", None], name="max_features"),
                Categorical(["balanced", "balanced_subsample"], name="class_weight"),
                Categorical(["gini", "entropy", "log_loss"], name="criterion")]

        self.random_state = random_state
        self.feature_space = None
        self.search_space = None

        self.num_features_per_sensor = None
        self.group_sizes = None

        self.X_train, self.X_val = None, None
        self.y_train, self.y_val = None, None

        self.best_feature_mask, self.best_params = None, None



    def construct_feature_mask(self, params):
        '''
        Function to construct the feature mask which is used in optimization
        '''
        return [1 if i in self.required_features else params.get(f"feature_{i}", 0)
                for i in range(self.num_features_per_sensor)]


    def select_features(self, X, feature_mask):
        '''
        Extract selected features based on binary mask used in optimization
        '''
        selected_indices = []
        start_idx = 0
        
        for i, include in enumerate(feature_mask):
            group_size = self.group_sizes[i]  
            if include == 1 or i in self.required_features:  # Always include required groups
                selected_indices.extend(range(start_idx, start_idx + group_size))
            start_idx += group_size  
            
        return X[:, selected_indices] if selected_indices else X  

            
    def do_bayes_opt(self, X_train, y_train, X_val, y_val):

        self.X_train, self.X_val = X_train, X_val
        self.y_train, self.y_val = y_train, y_val
            
        if self.method == 'PS10':
            num_total_features = len(self.X_train[0])
            self.num_features_per_sensor = int(num_total_features/10)
            self.group_sizes = [10] * self.num_features_per_sensor
        elif self.method == "PS1":
            self.num_features_per_sensor = len(self.X_train[0])
            self.group_sizes = [1] * self.num_features_per_sensor
        else:
            raise ValueError(f"Not a valid periodicity score method")
            
        if self.feature_selection:
            self.feature_space = [
                Integer(0, 1, name=f"feature_{i}") for i in range(self.num_features_per_sensor) if i not in self.required_features
            ]
            
            self.search_space = self.feature_space + self.hyperparameter_space
        else:
            self.search_space = self.hyperparameter_space

        @use_named_args(self.search_space)
        def objective(**params):
            if self.feature_selection:
                feature_mask = self.construct_feature_mask(params)
                hyperparams = {k: v for k, v in params.items() if k not in [f"feature_{i}" for i in range(self.num_features_per_sensor)]}
                X_train = self.select_features(self.X_train, feature_mask)
                X_val = self.select_features(self.X_val, feature_mask)
            else:
                X_train = self.X_train
                X_val = self.X_val
                hyperparams = params
                
            model = RandomForestClassifier(**hyperparams, random_state=42, n_jobs=-1)
            model.fit(X_train,self.y_train)
        
            y_val_pred = model.predict(X_val)
            val_f1 = f1_score(self.y_val, y_val_pred, average="macro")
            
            return -val_f1 
        # Perform Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.search_space,
            n_calls=self.n_calls, #Total number of iterations
            n_random_starts=self.n_random_starts,   # Random samples before using Bayesian optimization to build probabilistic model
            random_state=self.random_state)

        best_params = dict(zip([dim.name for dim in self.search_space], result.x))
        if self.feature_selection:
            best_feature_mask = self.construct_feature_mask(best_params)
            best_hyperparams = {k: v for k, v in best_params.items() if not k.startswith("feature_")}
            self.best_feature_mask = best_feature_mask
            self.best_params = best_hyperparams
        else:
            self.best_params = best_params




           
