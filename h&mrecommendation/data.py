

import pandas as pd


class DataLoader():

    def __init__(self):

        self.base_path = './input/h-and-m-personalized-fashion-recommendations/'
        self.csv_train = f'{self.base_path}transactions_train.csv'
        self.csv_sub = f'{self.base_path}sample_submission.csv'
        self.csv_users = f'{self.base_path}customers.csv'
        self.csv_items = f'{self.base_path}articles.csv'

    def load(self,df_name):

        if df_name == 'train':
            df = pd.read_csv(self.csv_train, dtype={'article_id': str}, parse_dates=['t_dat'] )
            return df
        elif df_name == 'sub':

            df_sub = pd.read_csv(self.csv_sub)
            return df_sub
        elif df_name == 'dfu':

            dfu = pd.read_csv(self.csv_users)
            return dfu
        elif df_name == 'dfi':

            dfi = pd.read_csv(self.csv_items, dtype={'article_id': str})
            return dfi
    @staticmethod
    ## Function to reduce the DF size
    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] =='int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col]= df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col]= df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col]= df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col]= df[col].astype(np.int8)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            end_mem = df.memory_usage().sum() / 1024**2
            if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
            return df