import base64
import io
import os

import numpy as np
import pandas as pd

import scipy.stats as stats
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, \
    confusion_matrix
import umap

from itertools import product
from pyhtml2pdf import converter


class GeneralUtils:
    @staticmethod
    def parse_contents(contents):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        dataset = pd.read_csv(io.StringIO(decoded))
        return dataset

    @staticmethod
    def scale_data(dataset):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(dataset)

        return pd.DataFrame(scaled, columns=dataset.columns.tolist())

    @staticmethod
    def convert_html_to_pdf(path_html, string_html, path_pdf):
        with open(path_html, 'w') as f:
            f.write(string_html)

        path = os.path.abspath(path_html)
        converter.convert(f'file:///{path}', path_pdf)


class ResemblanceMetrics:
    class URA:
        @staticmethod
        def ks_tests(real, synthetic):
            attribute_names = real.columns
            p_values = list()

            for c in attribute_names:
                _, p = stats.ks_2samp(real[c], synthetic[c])
                p_values.append(np.round(p, 5))

            dict_pvalues = dict(zip(attribute_names, p_values))

            return dict_pvalues

        @staticmethod
        def student_t_tests(real, synthetic):
            attribute_names = real.columns
            p_values = list()

            for c in attribute_names:
                _, p = stats.ttest_ind(real[c], synthetic[c])
                p_values.append(np.round(p, 5))

            dict_pvalues = dict(zip(attribute_names, p_values))

            return dict_pvalues

        @staticmethod
        def mann_whitney_tests(real, synthetic):
            attribute_names = real.columns
            p_values = list()

            for c in attribute_names:
                _, p = stats.mannwhitneyu(real[c], synthetic[c])
                p_values.append(np.round(p, 5))

            dict_pvalues = dict(zip(attribute_names, p_values))

            return dict_pvalues

        @staticmethod
        def chi_squared_tests(real, synthetic):
            attribute_names = real.columns
            p_values = list()

            for c in attribute_names:
                observed = pd.crosstab(real[c], synthetic[c])
                _, p, _, _ = stats.chi2_contingency(observed)
                p_values.append(np.round(p, 5))

            dict_pvalues = dict(zip(attribute_names, p_values))

            return dict_pvalues

        @staticmethod
        def cosine_distances(real, synthetic):
            attribute_names = real.columns
            distances = list()

            for c in attribute_names:
                distances.append(distance.cosine(real[c].values, synthetic[c].values))

            dict_distances = dict(zip(attribute_names, np.round(distances, 5)))

            return dict_distances

        @staticmethod
        def js_distances(real, synthetic):
            attribute_names = real.columns
            distances = list()

            for c in attribute_names:
                prob_distribution_real = stats.gaussian_kde(real[c].values).pdf(real[c].values)
                prob_distribution_synthetic = stats.gaussian_kde(real[c].values).pdf(synthetic[c].values)
                distances.append(distance.jensenshannon(prob_distribution_real, prob_distribution_synthetic))

            dict_distances = dict(zip(attribute_names, np.round(distances, 5)))

            return dict_distances

        @staticmethod
        def wass_distances(real, synthetic):
            attribute_names = real.columns
            distances = list()

            for c in attribute_names:
                distances.append(stats.wasserstein_distance(real[c].values, synthetic[c].values))

            dict_distances = dict(zip(attribute_names, np.round(distances, 5)))

            return dict_distances

    class MRA:
        @staticmethod
        def compute_correlations_matrices(dataset, type, features):
            if type == "corr_num":
                return dataset[features].corr(method='pearson')
            else:  # "corr_cat"
                factors_paired = list(product(dataset[features].columns, repeat=2))

                chi2 = []
                for f in factors_paired:
                    if f[0] != f[1]:
                        chitest = stats.chi2_contingency(pd.crosstab(dataset[f[0]], dataset[f[1]]))
                        chi2.append(chitest[0])
                    else:
                        chi2.append(0)

                chi2 = np.array(chi2).reshape((-1, len(features)))
                chi2 = pd.DataFrame(chi2, index=dataset[features].columns, columns=dataset[features].columns)

                return (chi2 - np.min(chi2, axis=None)) / np.ptp(chi2, axis=None)

        @staticmethod
        def check_lof(dataset):
            clf = LocalOutlierFactor(n_neighbors=2)

            labels_out = clf.fit_predict(dataset)
            neg_lof_score = clf.negative_outlier_factor_

            return neg_lof_score

        @staticmethod
        def do_pca(dataset):
            pca = PCA()
            pca.fit(dataset)
            var_ratio_cum = np.cumsum(pca.explained_variance_ratio_ * 100)

            return var_ratio_cum

        @staticmethod
        def do_umap(dataset, num_neighbors, min_dist):
            reducer = umap.UMAP(n_neighbors=num_neighbors, min_dist=min_dist, n_components=2)
            embedding = reducer.fit_transform(dataset)
            return embedding

    class DLA:
        @staticmethod
        def classify_real_vs_synthetic_data(models_names, real, synthetic, numeric_features, categorical_features):
            real["label"] = 0
            synthetic["label"] = 1

            combined_data = pd.concat([real, synthetic], ignore_index=True)

            train_data, test_data, train_labels, test_labels = train_test_split(combined_data.drop("label", axis=1),
                                                                                combined_data["label"], test_size=0.2)

            numeric_transformer = StandardScaler()

            data = pd.concat([train_data, test_data], ignore_index=True)
            categories_list = [np.unique(data[col]) for col in categorical_features]
            categorical_transformer = OneHotEncoder(categories=categories_list)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, numeric_features),
                    ("categorical", categorical_transformer, categorical_features)
                ])

            train_data_preprocessed = preprocessor.fit_transform(train_data)
            test_data_preprocessed = preprocessor.transform(test_data)

            classifiers = {'RF': RandomForestClassifier(n_estimators=100, random_state=9, n_jobs=3),
                           'KNN': KNeighborsClassifier(n_neighbors=9, n_jobs=3),
                           'DT': DecisionTreeClassifier(random_state=9),
                           'SVM': SVC(C=100, max_iter=300, kernel='linear', probability=True, random_state=9),
                           'MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=9)}

            results = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1'])

            for clas_name in models_names:
                classifiers[clas_name].fit(train_data_preprocessed, train_labels)

                predictions = classifiers[clas_name].predict(test_data_preprocessed)

                clas_results = pd.DataFrame([[clas_name,
                                              np.round(accuracy_score(test_labels, predictions), 4),
                                              np.round(precision_score(test_labels, predictions), 4),
                                              np.round(recall_score(test_labels, predictions), 4),
                                              np.round(f1_score(test_labels, predictions), 4)]],
                                            columns=results.columns)

                results = pd.concat([results, clas_results], ignore_index=True)

            return results


class UtilityMetrics:
    @staticmethod
    def train_test_model(model_name, train_data, test_data, train_labels, test_labels, numeric_features,
                         categorical_features):
        models = {
            "RF": RandomForestClassifier(n_estimators=100, n_jobs=3, random_state=10),
            "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=3),
            "DT": DecisionTreeClassifier(random_state=10),
            "SVM": SVC(C=100, max_iter=300, kernel='linear', probability=True, random_state=10),
            "MLP": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=10)
        }

        model = models[model_name]

        train_labels = train_labels.astype(str)
        test_labels = test_labels.astype(str)

        numeric_transformer = StandardScaler()
        data = pd.concat([train_data, test_data], ignore_index=True)
        categories_list = [np.unique(data[col]) for col in categorical_features]
        categorical_transformer = OneHotEncoder(categories=categories_list)

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical", categorical_transformer, categorical_features)
            ])

        train_data_preprocessed = preprocessor.fit_transform(train_data)
        test_data_preprocessed = preprocessor.transform(test_data)

        model.fit(train_data_preprocessed, train_labels)
        predictions = model.predict(test_data_preprocessed)

        results = pd.DataFrame([[model_name,
                                 np.round(accuracy_score(test_labels, predictions), 4),
                                 np.round(precision_score(test_labels, predictions, average=None)[-1], 4),
                                 np.round(recall_score(test_labels, predictions, average=None)[-1], 4),
                                 np.round(f1_score(test_labels, predictions, average=None)[-1], 4)]],
                               columns=['model', 'accuracy', 'precision', 'recall', 'f1'])

        return results, confusion_matrix(test_labels, predictions)


class PrivacyMetrics:
    class SEA:
        @staticmethod
        def pairwise_euclidean_distance(real, synthetic):
            r = GeneralUtils.scale_data(real)
            s = GeneralUtils.scale_data(synthetic)
            distances = distance.cdist(r, s, 'euclidean')

            return np.round(distances, 4)

        @staticmethod
        def str_similarity(real, synthetic):
            distances = cosine_similarity(real, synthetic)

            return np.round(distances, 4)

        @staticmethod
        def hausdorff_distance(real, synthetic):
            r = GeneralUtils.scale_data(real)
            s = GeneralUtils.scale_data(synthetic)
            distances = max(distance.directed_hausdorff(r, s)[0], distance.directed_hausdorff(s, r)[0])

            return np.round(distances, 4)

    class MIA:
        @staticmethod
        def simulate_mia(real_subset_attacker, label_membership_train, synthetic, threshold):
            distances = cosine_similarity(real_subset_attacker, synthetic)

            records_identified = (distances > threshold).any(axis=1)  # if TRUE, the real row is identified

            precision_attacker = precision_score(label_membership_train, records_identified)
            accuracy_attacker = accuracy_score(label_membership_train, records_identified)

            return precision_attacker, accuracy_attacker

    class AIA:
        @staticmethod
        def simulate_aia(real, synthetic, QID_features_names, target_features_names, dict_type):
            real_subset_attacker = real[QID_features_names]

            train_synthetic_data_QID = synthetic[QID_features_names]

            train_features_type = {key: dict_type[key] for key in QID_features_names}

            train_numeric_features = [key for key, value in train_features_type.items() if value == "numerical"]
            train_categorical_features = [key for key, value in train_features_type.items() if value == "categorical"]

            numeric_transformer = StandardScaler()
            data = pd.concat([train_synthetic_data_QID, real_subset_attacker], ignore_index=True)
            categories_list = [np.unique(data[col]) for col in train_categorical_features]
            categorical_transformer = OneHotEncoder(categories=categories_list)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, train_numeric_features),
                    ("categorical", categorical_transformer, train_categorical_features)
                ])

            train_data_preprocessed = preprocessor.fit_transform(train_synthetic_data_QID)
            test_data_preprocessed = preprocessor.transform(real_subset_attacker)

            results = list()

            for target_name in target_features_names:

                if dict_type[target_name] == "numerical":

                    model = DecisionTreeRegressor(random_state=23)
                    model.fit(train_data_preprocessed, synthetic[target_name])
                    predictions = model.predict(test_data_preprocessed)
                    results.append(
                        ['rmse', target_name,
                         np.round(mean_squared_error(real[target_name], predictions, squared=False), 4),
                         str([np.round(np.percentile(real[target_name], 25), 4),
                              np.round(np.percentile(real[target_name], 75), 4)])])

                else:

                    model = DecisionTreeClassifier(random_state=23)
                    model.fit(train_data_preprocessed, synthetic[target_name].astype(str))
                    predictions = model.predict(test_data_preprocessed)
                    results.append(
                        ['acc', target_name, np.round(accuracy_score(real[target_name].astype(str), predictions), 4),
                         []])

            return pd.DataFrame(results, columns=['Metric name', 'Target name', 'Value', 'IQR target'])
