import os
import gc
import random
import sys
import logging
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
import category_encoders as ce
import multiprocessing as mp


from tqdm import tqdm
from copy import deepcopy
from attrdict import AttrDict
from steppy.base import BaseTransformer
from steppy.utils import get_logger
from functools import partial
from functools import reduce
from scipy.stats import kurtosis, iqr, skew
from sklearn.linear_model import LinearRegression
from .utils import parallel_apply, safe_div,set_seed
from sklearn.externals import joblib



logger = get_logger()


def create_submission(meta, predictions):
    submission = pd.DataFrame({'SK_ID_CURR': meta['SK_ID_CURR'].tolist(),
                               'TARGET': predictions
                               })
    return submission


def verify_submission(submission, sample_submission):
    assert submission.shape == sample_submission.shape, \
        'Expected submission to have shape {} but got {}'.format(sample_submission.shape, submission.shape)

    for submission_id, correct_id in zip(submission['SK_ID_CURR'].values, sample_submission['SK_ID_CURR'].values):
        assert correct_id == submission_id, \
            'Wrong id: expected {} but got {}'.format(correct_id, submission_id)


def get_logger():
    return logging.getLogger('home-credit')


def init_logger():
    logger = logging.getLogger('home-credit')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger





# def read_yaml(filepath):
#     with open(filepath) as f:
#         config = yaml.load(f)
#     return AttrDict(config)


# def parameter_eval(param):
#     try:
#         return eval(param)
#     except Exception:
#         return param


def persist_evaluation_predictions(experiment_directory, y_pred, raw_data, id_column, target_column):
    raw_data.loc[:, 'y_pred'] = y_pred.reshape(-1)
    predictions_df = raw_data.loc[:, [id_column, target_column, 'y_pred']]
    filepath = os.path.join(experiment_directory, 'evaluation_predictions.csv')
    logging.info('evaluation predictions csv shape: {}'.format(predictions_df.shape))
    predictions_df.to_csv(filepath, index=None)


def set_seed(seed=90210):
    random.seed(seed)
    np.random.seed(seed)


def calculate_rank(predictions):
    rank = (1 + predictions.rank().values) / (predictions.shape[0] + 1)
    return rank


def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def read_oof_predictions(prediction_dir, train_filepath, id_column, target_column):
    labels = pd.read_csv(train_filepath, usecols=[id_column, target_column])

    filepaths_train, filepaths_test = [], []
    for filepath in sorted(glob.glob('{}/*'.format(prediction_dir))):
        if filepath.endswith('_oof_train.csv'):
            filepaths_train.append(filepath)
        elif filepath.endswith('_oof_test.csv'):
            filepaths_test.append(filepath)

    train_dfs = []
    for filepath in filepaths_train:
        train_dfs.append(pd.read_csv(filepath))
    train_dfs = reduce(lambda df1, df2: pd.merge(df1, df2, on=[id_column, 'fold_id']), train_dfs)
    train_dfs.columns = _clean_columns(train_dfs, keep_colnames=[id_column, 'fold_id'])
    train_dfs = pd.merge(train_dfs, labels, on=[id_column])

    test_dfs = []
    for filepath in filepaths_test:
        test_dfs.append(pd.read_csv(filepath))
    test_dfs = reduce(lambda df1, df2: pd.merge(df1, df2, on=[id_column, 'fold_id']), test_dfs)
    test_dfs.columns = _clean_columns(test_dfs, keep_colnames=[id_column, 'fold_id'])
    return train_dfs, test_dfs


def _clean_columns(df, keep_colnames):
    new_colnames = keep_colnames
    feature_colnames = df.drop(keep_colnames, axis=1).columns
    for i, colname in enumerate(feature_colnames):
        new_colnames.append('model_{}'.format(i))
    return new_colnames

def safe_div(a, b):
    try:
        return float(a) / float(b)
    except:
        return 0.0


logger = get_logger()

class ApplicationCleaning(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, X):
        X['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        X['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        X['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        X['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

        return {'X': X}


class BureauCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, bureau):
        bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
        bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
        bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan

        if self.fill_missing:
            bureau['AMT_CREDIT_SUM'].fillna(self.fill_value, inplace=True)
            bureau['AMT_CREDIT_SUM_DEBT'].fillna(self.fill_value, inplace=True)
            bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(self.fill_value, inplace=True)
            bureau['CNT_CREDIT_PROLONG'].fillna(self.fill_value, inplace=True)

        return {'bureau': bureau}


class CreditCardCleaning(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, credit_card):
        credit_card['AMT_DRAWINGS_ATM_CURRENT'][credit_card['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
        credit_card['AMT_DRAWINGS_CURRENT'][credit_card['AMT_DRAWINGS_CURRENT'] < 0] = np.nan

        return {'credit_card': credit_card}


class PreviousApplicationCleaning(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, previous_application):
        previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

        return {'previous_application': previous_application}




#############################################feature_extraction############################################################
class FeatureJoiner(BaseTransformer):
    def __init__(self, use_nan_count=False, **kwargs):
        super().__init__()
        self.use_nan_count = use_nan_count

    def transform(self, numerical_feature_list, categorical_feature_list, **kwargs):
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature.reset_index(drop=True, inplace=True)
        features = pd.concat(features, axis=1).astype(np.float32)
        if self.use_nan_count:
            features['nan_count'] = features.isnull().sum(axis=1)

        outputs = dict()
        outputs['features'] = features
        outputs['feature_names'] = list(features.columns)
        outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        return outputs

    def _get_feature_names(self, dataframes):
        feature_names = []
        for dataframe in dataframes:
            try:
                feature_names.extend(list(dataframe.columns))
            except Exception as e:
                print(e)
                feature_names.append(dataframe.name)

        return feature_names


class CategoricalEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.categorical_columns = kwargs['categorical_columns']
        params = deepcopy(kwargs)
        params.pop('categorical_columns', None)
        self.params = params
        self.encoder_class = ce.OrdinalEncoder
        self.categorical_encoder = None

    def fit(self, X, y, **kwargs):
        X_ = X[self.categorical_columns]
        self.categorical_encoder = self.encoder_class(cols=self.categorical_columns, **self.params)
        self.categorical_encoder.fit(X_, y)
        return self

    def transform(self, X, **kwargs):
        X_ = X[self.categorical_columns]
        X_ = self.categorical_encoder.transform(X_)
        return {'categorical_features': X_}

    def load(self, filepath):
        self.categorical_encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.categorical_encoder, filepath)


class GroupbyAggregateDiffs(BaseTransformer):
    def __init__(self, groupby_aggregations, use_diffs_only=False, **kwargs):
        super().__init__()
        self.groupby_aggregations = groupby_aggregations
        self.use_diffs_only = use_diffs_only
        self.features = []
        self.groupby_feature_names = []

    @property
    def feature_names(self):
        if self.use_diffs_only:
            return self.diff_feature_names
        else:
            return self.groupby_feature_names + self.diff_feature_names

    def fit(self, main_table, **kwargs):
        for groupby_cols, specs in self.groupby_aggregations:
            group_object = main_table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                group_features = group_object[select].agg(agg).reset_index() \
                    .rename(index=str,
                            columns={select: groupby_aggregate_name})[groupby_cols + [groupby_aggregate_name]]

                self.features.append((groupby_cols, group_features))
                self.groupby_feature_names.append(groupby_aggregate_name)
        return self

    def transform(self, main_table, **kwargs):
        main_table = self._merge_grouby_features(main_table)
        main_table = self._add_diff_features(main_table)

        return {'numerical_features': main_table[self.feature_names].astype(np.float32)}

    def _merge_grouby_features(self, main_table):
        for groupby_cols, groupby_features in self.features:
            main_table = main_table.merge(groupby_features,
                                          on=groupby_cols,
                                          how='left')
        return main_table

    def _add_diff_features(self, main_table):
        self.diff_feature_names = []
        for groupby_cols, specs in self.groupby_aggregations:
            for select, agg in specs:
                if agg in ['mean', 'median', 'max', 'min']:
                    groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                    diff_feature_name = '{}_diff'.format(groupby_aggregate_name)
                    abs_diff_feature_name = '{}_abs_diff'.format(groupby_aggregate_name)

                    main_table[diff_feature_name] = main_table[select] - main_table[groupby_aggregate_name]
                    main_table[abs_diff_feature_name] = np.abs(main_table[select] - main_table[groupby_aggregate_name])

                    self.diff_feature_names.append(diff_feature_name)
                    self.diff_feature_names.append(abs_diff_feature_name)

        return main_table

    def load(self, filepath):
        params = joblib.load(filepath)
        self.features = params['features']
        self.groupby_feature_names = params['groupby_feature_names']
        return self

    def persist(self, filepath):
        params = {'features': self.features,
                  'groupby_feature_names': self.groupby_feature_names}
        joblib.dump(params, filepath)

    def _create_colname_from_specs(self, groupby_cols, agg, select):
        return '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)


class GroupbyAggregate(BaseTransformer):
    def __init__(self, table_name, id_columns, groupby_aggregations, **kwargs):
        super().__init__()
        self.table_name = table_name
        self.id_columns = id_columns
        self.groupby_aggregations = groupby_aggregations

    def fit(self, table, **kwargs):
        features = pd.DataFrame({self.id_columns[0]: table[self.id_columns[0]].unique()})

        for groupby_cols, specs in self.groupby_aggregations:
            group_object = table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                features = features.merge(group_object[select]
                                          .agg(agg)
                                          .reset_index()
                                          .rename(index=str,
                                                  columns={select: groupby_aggregate_name})
                                          [groupby_cols + [groupby_aggregate_name]],
                                          on=groupby_cols,
                                          how='left')
        self.features = features
        return self

    def transform(self, table, **kwargs):
        return {'features_table': self.features}

    def load(self, filepath):
        self.features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.features, filepath)

    def _create_colname_from_specs(self, groupby_cols, select, agg):
        return '{}_{}_{}_{}'.format(self.table_name, '_'.join(groupby_cols), agg, select)


class GroupbyMerge(BaseTransformer):
    def __init__(self, id_columns, **kwargs):
        super().__init__()
        self.id_columns = id_columns

    def _feature_names(self, features):
        feature_names = list(features.columns)
        feature_names.remove(self.id_columns[0])
        return feature_names

    def transform(self, table, features, **kwargs):
        table = table.merge(features,
                            left_on=[self.id_columns[0]],
                            right_on=[self.id_columns[1]],
                            how='left',
                            validate='one_to_one')

        return {'numerical_features': table[self._feature_names(features)].astype(np.float32)}


class BasicHandCraftedFeatures(BaseTransformer):
    def __init__(self, num_workers=1, **kwargs):
        self.num_workers = num_workers
        self.features = None

    @property
    def feature_names(self):
        feature_names = list(self.features.columns)
        feature_names.remove('SK_ID_CURR')
        return feature_names

    def transform(self, **kwargs):
        return {'features_table': self.features}

    def load(self, filepath):
        self.features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.features, filepath)


class ApplicationFeatures(BaseTransformer):
    def __init__(self, categorical_columns, numerical_columns):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.engineered_numerical_columns = ['annuity_income_percentage',
                                             'car_to_birth_ratio',
                                             'car_to_employ_ratio',
                                             'children_ratio',
                                             'credit_to_annuity_ratio',
                                             'credit_to_goods_ratio',
                                             'credit_to_income_ratio',
                                             'days_employed_percentage',
                                             'income_credit_percentage',
                                             'income_per_child',
                                             'income_per_person',
                                             'payment_rate',
                                             'phone_to_birth_ratio',
                                             'phone_to_employ_ratio',
                                             'external_sources_weighted',
                                             'external_sources_min',
                                             'external_sources_max',
                                             'external_sources_sum',
                                             'external_sources_mean',
                                             'external_sources_nanmedian',
                                             'short_employment',
                                             'young_age',
                                             'cnt_non_child',
                                             'child_to_non_child_ratio',
                                             'income_per_non_child',
                                             'credit_per_person',
                                             'credit_per_child',
                                             'credit_per_non_child',
                                             ]

    def transform(self, X, **kwargs):
        X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']
        X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
        X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
        X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
        X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
        X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
        X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
        X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
        X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
        X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']
        X['external_sources_weighted'] = X.EXT_SOURCE_1 * 2 + X.EXT_SOURCE_2 * 3 + X.EXT_SOURCE_3 * 4
        X['cnt_non_child'] = X['CNT_FAM_MEMBERS'] - X['CNT_CHILDREN']
        X['child_to_non_child_ratio'] = X['CNT_CHILDREN'] / X['cnt_non_child']
        X['income_per_non_child'] = X['AMT_INCOME_TOTAL'] / X['cnt_non_child']
        X['credit_per_person'] = X['AMT_CREDIT'] / X['CNT_FAM_MEMBERS']
        X['credit_per_child'] = X['AMT_CREDIT'] / (1 + X['CNT_CHILDREN'])
        X['credit_per_non_child'] = X['AMT_CREDIT'] / X['cnt_non_child']
        for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            X['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        X['short_employment'] = (X['DAYS_EMPLOYED'] < -2000).astype(int)
        X['young_age'] = (X['DAYS_BIRTH'] < -14000).astype(int)

        return {'numerical_features': X[self.engineered_numerical_columns + self.numerical_columns],
                'categorical_features': X[self.categorical_columns]
                }


class BureauFeatures(BasicHandCraftedFeatures):
    def fit(self, bureau, **kwargs):
        bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
        bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
        features = pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].unique()})

        groupby = bureau.groupby(by=['SK_ID_CURR'])

        g = groupby['DAYS_CREDIT'].agg('count').reset_index()
        g.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['CREDIT_TYPE'].agg('nunique').reset_index()
        g.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['bureau_credit_active_binary'].agg('mean').reset_index()
        g.rename(index=str, columns={'bureau_credit_active_binary': 'bureau_credit_active_binary'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
        g.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['bureau_credit_enddate_binary'].agg('mean').reset_index()
        g.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        features['bureau_average_of_past_loans_per_type'] = \
            features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']

        features['bureau_debt_credit_ratio'] = \
            features['bureau_total_customer_debt'] / features['bureau_total_customer_credit']

        features['bureau_overdue_debt_ratio'] = \
            features['bureau_total_customer_overdue'] / features['bureau_total_customer_debt']

        self.features = features
        return self


class CreditCardBalanceFeatures(BasicHandCraftedFeatures):
    def fit(self, credit_card, **kwargs):
        static_features = self._static_features(credit_card, **kwargs)
        dynamic_features = self._dynamic_features(credit_card, **kwargs)

        self.features = pd.merge(static_features,
                                 dynamic_features,
                                 on=['SK_ID_CURR'],
                                 validate='one_to_one')
        return self

    def _static_features(self, credit_card, **kwargs):
        credit_card['number_of_installments'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
            'CNT_INSTALMENT_MATURE_CUM']

        credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
            lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]

        features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        groupby = credit_card.groupby(by=['SK_ID_CURR'])

        g = groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['SK_DPD'].agg('mean').reset_index()
        g.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['number_of_installments'].agg('sum').reset_index()
        g.rename(index=str, columns={'number_of_installments': 'credit_card_total_installments'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
        g.rename(index=str,
                 columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features[
            'credit_card_drawings_total']

        features['credit_card_installments_per_loan'] = (
            features['credit_card_total_installments'] / features['credit_card_number_of_loans'])

        return features

    def _dynamic_features(self, credit_card, **kwargs):
        features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        credit_card_sorted = credit_card.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

        groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])
        credit_card_sorted['credit_card_monthly_diff'] = groupby['AMT_BALANCE'].diff()
        groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])

        g = groupby['credit_card_monthly_diff'].agg('mean').reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        return features


class POSCASHBalanceFeatures(BasicHandCraftedFeatures):
    def __init__(self, last_k_agg_periods, last_k_trend_periods, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.last_k_agg_periods = last_k_agg_periods
        self.last_k_trend_periods = last_k_trend_periods

        self.num_workers = num_workers
        self.features = None

    def fit(self, pos_cash, **kwargs):
        pos_cash['is_contract_status_completed'] = pos_cash['NAME_CONTRACT_STATUS'] == 'Completed'
        pos_cash['pos_cash_paid_late'] = (pos_cash['SK_DPD'] > 0).astype(int)
        pos_cash['pos_cash_paid_late_with_tolerance'] = (pos_cash['SK_DPD_DEF'] > 0).astype(int)

        features = pd.DataFrame({'SK_ID_CURR': pos_cash['SK_ID_CURR'].unique()})
        groupby = pos_cash.groupby(['SK_ID_CURR'])
        func = partial(POSCASHBalanceFeatures.generate_features,
                       agg_periods=self.last_k_agg_periods,
                       trend_periods=self.last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=self.num_workers).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        self.features = features
        return self

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods):
        one_time = POSCASHBalanceFeatures.one_time_features(gr)
        all = POSCASHBalanceFeatures.all_installment_features(gr)
        agg = POSCASHBalanceFeatures.last_k_installment_features(gr, agg_periods)
        trend = POSCASHBalanceFeatures.trend_in_last_k_installment_features(gr, trend_periods)
        last = POSCASHBalanceFeatures.last_loan_features(gr)
        features = {**one_time, **all, **agg, **trend, **last}
        return pd.Series(features)

    @staticmethod
    def one_time_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], inplace=True)
        features = {}

        features['pos_cash_remaining_installments'] = gr_['CNT_INSTALMENT_FUTURE'].tail(1)
        features['pos_cash_completed_contracts'] = gr_['is_contract_status_completed'].agg('sum')

        return features

    @staticmethod
    def all_installment_features(gr):
        return POSCASHBalanceFeatures.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_.iloc[:period]

            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late_with_tolerance',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD_DEF',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_.iloc[:period]

            features = add_trend_feature(features, gr_period,
                                         'SK_DPD', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'SK_DPD_DEF', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'CNT_INSTALMENT_FUTURE', '{}_period_trend_'.format(period)
                                         )
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
        last_installment_id = gr_['SK_ID_PREV'].iloc[0]
        gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

        features={}
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late',
                                         ['count', 'sum', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late_with_tolerance',
                                         ['mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD',
                                         ['sum', 'mean', 'max', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD_DEF',
                                         ['sum', 'mean', 'max', 'std'],
                                         'last_loan_')

        return features


class PreviousApplicationFeatures(BasicHandCraftedFeatures):
    def __init__(self, numbers_of_applications=[], num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.numbers_of_applications = numbers_of_applications

    def fit(self, prev_applications, **kwargs):
        features = pd.DataFrame({'SK_ID_CURR': prev_applications['SK_ID_CURR'].unique()})

        prev_app_sorted = prev_applications.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
        prev_app_sorted_groupby = prev_app_sorted.groupby(by=['SK_ID_CURR'])

        prev_app_sorted['previous_application_prev_was_approved'] = (
            prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
        g = prev_app_sorted_groupby['previous_application_prev_was_approved'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        prev_app_sorted['previous_application_prev_was_refused'] = (
            prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
        g = prev_app_sorted_groupby['previous_application_prev_was_refused'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted_groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted.groupby(by=['SK_ID_CURR'])['previous_application_prev_was_refused'].mean().reset_index()
        g.rename(index=str, columns={
            'previous_application_prev_was_refused': 'previous_application_fraction_of_refused_applications'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        prev_app_sorted['prev_applications_prev_was_revolving_loan'] = (
            prev_app_sorted['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int')
        g = prev_app_sorted.groupby(by=['SK_ID_CURR'])[
            'prev_applications_prev_was_revolving_loan'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        for number in self.numbers_of_applications:
            prev_applications_tail = prev_app_sorted_groupby.tail(number)

            tail_groupby = prev_applications_tail.groupby(by=['SK_ID_CURR'])

            g = tail_groupby['CNT_PAYMENT'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

            g = tail_groupby['DAYS_DECISION'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(
                         number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

            g = tail_groupby['DAYS_FIRST_DRAWING'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={
                         'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(
                             number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

        self.features = features
        return self


class InstallmentPaymentsFeatures(BasicHandCraftedFeatures):
    def __init__(self, last_k_agg_periods, last_k_agg_period_fractions, last_k_trend_periods, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.last_k_agg_periods = last_k_agg_periods
        self.last_k_agg_period_fractions = last_k_agg_period_fractions
        self.last_k_trend_periods = last_k_trend_periods

        self.num_workers = num_workers
        self.features = None

    def fit(self, installments, **kwargs):
        installments['installment_paid_late_in_days'] = installments['DAYS_ENTRY_PAYMENT'] - installments[
            'DAYS_INSTALMENT']
        installments['installment_paid_late'] = (installments['installment_paid_late_in_days'] > 0).astype(int)
        installments['installment_paid_over_amount'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
        installments['installment_paid_over'] = (installments['installment_paid_over_amount'] > 0).astype(int)

        features = pd.DataFrame({'SK_ID_CURR': installments['SK_ID_CURR'].unique()})
        groupby = installments.groupby(['SK_ID_CURR'])

        func = partial(InstallmentPaymentsFeatures.generate_features,
                       agg_periods=self.last_k_agg_periods,
                       period_fractions=self.last_k_agg_period_fractions,
                       trend_periods=self.last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=self.num_workers).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        self.features = features
        return self

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods, period_fractions):
        all = InstallmentPaymentsFeatures.all_installment_features(gr)
        agg = InstallmentPaymentsFeatures.last_k_installment_features_with_fractions(gr,
                                                                                     agg_periods,
                                                                                     period_fractions)
        trend = InstallmentPaymentsFeatures.trend_in_last_k_installment_features(gr, trend_periods)
        last = InstallmentPaymentsFeatures.last_loan_features(gr)
        features = {**all, **agg, **trend, **last}
        return pd.Series(features)

    @staticmethod
    def all_installment_features(gr):
        return InstallmentPaymentsFeatures.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features_with_fractions(gr, periods, period_fractions):
        features = InstallmentPaymentsFeatures.last_k_installment_features(gr, periods)

        for short_period, long_period in period_fractions:
            short_feature_names = get_feature_names_by_period(features, short_period)
            long_feature_names = get_feature_names_by_period(features, long_period)

            for short_feature, long_feature in zip(short_feature_names, long_feature_names):
                old_name_chunk = '_{}_'.format(short_period)
                new_name_chunk = '_{}by{}_fraction_'.format(short_period, long_period)
                fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
                features[fraction_feature_name] = safe_div(features[short_feature], features[long_feature])
        return features

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_.iloc[:period]

            features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                             period_name)

            features = add_features_in_group(features, gr_period, 'installment_paid_late_in_days',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_over_amount',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_over',
                                             ['count', 'mean'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_.iloc[:period]

            features = add_trend_feature(features, gr_period,
                                         'installment_paid_late_in_days', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'installment_paid_over_amount', '{}_period_trend_'.format(period)
                                         )
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
        last_installment_id = gr_['SK_ID_PREV'].iloc[0]
        gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

        features = {}
        features = add_features_in_group(features, gr_,
                                         'installment_paid_late_in_days',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_late',
                                         ['count', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_over_amount',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_over',
                                         ['count', 'mean'],
                                         'last_loan_')
        return features


class ConcatFeatures(BaseTransformer):
    def transform(self, **kwargs):
        features_concat = []
        for _, feature in kwargs.items():
            feature.reset_index(drop=True, inplace=True)
            features_concat.append(feature)
        features_concat = pd.concat(features_concat, axis=1)
        return {'concatenated_features': features_concat}


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()

    return features


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def get_feature_names_by_period(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])


###########################################hyperparameter_tuning###########################################################
class RandomSearchOptimizer(BaseTransformer):
    def __init__(self,
                 TransformerClass,
                 params,
                 score_func,
                 maximize,
                 train_input_keys,
                 valid_input_keys,
                 n_runs,
                 callbacks=None):
        super().__init__()
        self.TransformerClass = TransformerClass
        self.param_space = create_param_space(params, n_runs)
        self.train_input_keys = train_input_keys
        self.valid_input_keys = valid_input_keys
        self.score_func = score_func
        self.maximize = maximize
        self.callbacks = callbacks or []
        self.best_transformer = TransformerClass(**self.param_space[0])

    def fit(self, **kwargs):
        if self.train_input_keys:
            train_inputs = {input_key: kwargs[input_key] for input_key in self.train_input_keys}
        else:
            train_inputs = kwargs
        X_valid, y_valid = kwargs[self.valid_input_keys[0]], kwargs[self.valid_input_keys[1]]

        results = []
        for i, param_set in enumerate(self.param_space):
            logger.info('training run {}'.format(i))
            logger.info('parameters: {}'.format(str(param_set)))
            transformer = self.TransformerClass(**param_set)
            transformer.fit(**train_inputs)

            y_pred_valid = transformer.transform(X_valid)
            y_pred_valid_value = list(y_pred_valid.values())[0]
            run_score = self.score_func(y_valid, y_pred_valid_value)
            results.append((run_score, param_set))

            del y_pred_valid, transformer
            gc.collect()

            for callback in self.callbacks:
                callback.on_run_end(score=run_score, params=param_set)

        assert len(results) > 0, 'All random search runs failed, check your parameter space'
        results_sorted = sorted(results, key=lambda x: x[0])

        if self.maximize:
            best_score, best_param_set = results_sorted[-1]
        else:
            best_score, best_param_set = results_sorted[0]

        for callback in self.callbacks:
            callback.on_search_end(results=results)

        self.best_transformer = self.TransformerClass(**best_param_set)
        self.best_transformer.fit(**train_inputs)
        return self

    def transform(self, **kwargs):
        return self.best_transformer.transform(**kwargs)

    def persist(self, filepath):
        self.best_transformer.persist(filepath)

    def load(self, filepath):
        self.best_transformer.load(filepath)
        return self


def create_param_space(params, n_runs):
    seed = np.random.randint(1000)
    param_space = []
    for i in range(n_runs):
        set_seed(seed + i)
        param_choice = {}
        for param, value in params.items():
            if isinstance(value, list):
                if len(value) == 2:
                    mode = 'choice'
                    param_choice[param] = sample_param_space(value, mode)
                else:
                    mode = value[-1]
                    param_choice[param] = sample_param_space(value[:-1], mode)
            else:
                param_choice[param] = value
        param_space.append(param_choice)
    set_seed()
    return param_space


def sample_param_space(value_range, mode):
    if mode == 'list':
        value = np.random.choice(value_range)
        if isinstance(value, np.str_):
            value = str(value)
    else:
        range_min, range_max = value_range
        if mode == 'choice':
            value = np.random.choice(range(range_min, range_max, 1))
        elif mode == 'uniform':
            value = np.random.uniform(low=range_min, high=range_max)
        elif mode == 'log-uniform':
            value = np.exp(np.random.uniform(low=np.log(range_min), high=np.log(range_max)))
        else:
            raise NotImplementedError
    return value


class GridSearchCallback:
    def on_run_end(self, score, params):
        return NotImplementedError

    def on_search_end(self, results):
        return NotImplementedError


############################################models##########################################################
class LightGBM(BaseTransformer):
    def __init__(self, name=None, **params):
        super().__init__()
        logger.info('initializing LightGBM...')
        self.params = params
        self.training_params = ['number_boosting_rounds', 'early_stopping_rounds']
        self.evaluation_function = None
        self.callbacks = callbacks(channel_prefix=name)

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self,
            X,
            y,
            X_valid,
            y_valid,
            feature_names='auto',
            categorical_features='auto',
            **kwargs):
        evaluation_results = {}

        self._check_target_shape_and_type(y, 'y')
        self._check_target_shape_and_type(y_valid, 'y_valid')
        y = self._format_target(y)
        y_valid = self._format_target(y_valid)

        logger.info('LightGBM, train data shape        {}'.format(X.shape))
        logger.info('LightGBM, validation data shape   {}'.format(X_valid.shape))
        logger.info('LightGBM, train labels shape      {}'.format(y.shape))
        logger.info('LightGBM, validation labels shape {}'.format(y_valid.shape))

        data_train = lgb.Dataset(data=X,
                                 label=y,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)
        data_valid = lgb.Dataset(X_valid,
                                 label=y_valid,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)

        self.estimator = lgb.train(self.model_config,
                                   data_train,
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   valid_sets=[data_train, data_valid],
                                   valid_names=['data_train', 'data_valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function,
                                   callbacks=self.callbacks,
                                   **kwargs)
        return self

    def transform(self, X, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)

    def _check_target_shape_and_type(self, target, name):
        if not any([isinstance(target, obj_type) for obj_type in [pd.Series, np.ndarray, list]]):
            raise TypeError(
                '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(target)))
        try:
            assert len(target.shape) == 1, '"{}" must be 1-D. It is {}-D instead.'.format(name,
                                                                                          len(target.shape))
        except AttributeError:
            print('Cannot determine shape of the {}. '
                  'Type must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead'.format(name,
                                                                                                     type(target)))

    def _format_target(self, target):

        if isinstance(target, pd.Series):
            return target.values
        elif isinstance(target, np.ndarray):
            return target
        elif isinstance(target, list):
            return np.array(target)
        else:
            raise TypeError(
                '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(target)))
