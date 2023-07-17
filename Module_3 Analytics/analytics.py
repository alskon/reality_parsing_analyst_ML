import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats


def groupby_columns(df, column, count=True):
    if count:
        table = df.groupby(['prj_id']).agg(count=('flat_id', 'count'), mean=(column, 'mean'),
                                            stdev=(column, 'std'), max=(column, 'max'),
                                            min=(column, 'min'), median=(column, 'median'),
                                            iq75_25=(column, lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25))
                                            ).reset_index()
    else:
        table = df.groupby(['prj_id']).agg(mean=(column, 'mean'),
                                            stdev=(column, 'std'), max=(column, 'max'),
                                            min=(column, 'min'), median=(column, 'median'),
                                            iq75_25=(column, lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25))
                                            ).reset_index()
    table['range'] = table['max'] - table['min']
    return table


def rename_df(df, name, r=None):
    new_name = f'{name}'
    if r:
        new_name = f'{name}_r_{r}'
    new_columns = {col: f'{new_name}_{col}' for col in df.columns if col != 'prj_id'}
    return df.rename(columns=new_columns)


def agg_dataframe(df):
    prj_id = pd.unique(df['prj_id'])
    met = pd.DataFrame(prj_id, columns=['prj_id'])
    rooms = [r for r in range(df['rooms'].min(), df['rooms'].max() + 1)]
    rooms.insert(0, None)
    names = ['all', 'all_m2', 'all_with_wb', 'all_wo_wb', 'all_with_wb_m2', 'all_wo_wb_m2']

    for r in rooms:
        dataframes = []
        if not r:
            dataframes.append(groupby_columns(df, 'today_price'))
            dataframes.append(groupby_columns(df, 'price_m2', False))
            dataframes.append(groupby_columns(df[df['whitebox'] == True], 'today_price'))
            dataframes.append(groupby_columns(df[df['whitebox'] == False], 'today_price'))
            dataframes.append(groupby_columns(df[df['whitebox'] == True], 'price_m2', False))
            dataframes.append(groupby_columns(df[df['whitebox'] == False], 'price_m2', False))
        else:
            dataframes.append(groupby_columns(df.loc[df['rooms'] == r], 'today_price'))
            dataframes.append(groupby_columns(df.loc[df['rooms'] == r], 'price_m2', False))
            dataframes.append(groupby_columns(df.loc[(df['whitebox'] == True) & (df['rooms'] == r)], 'today_price'))
            dataframes.append(groupby_columns(df.loc[(df['whitebox'] == False) & (df['rooms'] == r)], 'today_price'))
            dataframes.append(groupby_columns(df.loc[(df['whitebox'] == True) & (df['rooms'] == r)], 'price_m2', False))
            dataframes.append(groupby_columns(df.loc[(df['whitebox'] == False) & (df['rooms'] == r)], 'price_m2', False))
        renamed_df = [rename_df(d, name, r) for d, name in zip(dataframes, names)]
        for d in renamed_df:
            met = met.merge(d, 'outer', on='prj_id')
    return met


def calculate_metrics(flats):
    flats = pd.DataFrame(flats)
    flats['price_m2'] = np.round(flats['today_price'] / flats['area'], 2)
    #flats['change_price_today_prev'] = np.round(1 - (flats['today_price'] / flats['prev_price']), 2)
    #flats['change_price_today_first'] = np.round(1 - (flats['today_price'] / flats['first_price']), 2)
    table_met = agg_dataframe(flats)
    columns = table_met.columns.tolist()
    met_res = []
    for t in table_met.values:
        d = {}
        for i in range(0, len(t)):
            d[columns[i]] = t[i]
        met_res.append(d)
    return met_res


def visualisation_project(prj):
    print(prj)
    mu, sigma = prj['all_m2_mean'], prj['all_m2_stdev']
    x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()
    print(mu, sigma)


class Analytics:
    def __init__(self, prj, flats):
        self.projects = pd.DataFrame(prj)
        self.flats = pd.DataFrame(flats)
        self.project_metrics = None

    def calculate_prj_metrics(self):
        if self.flats is not None:
            for col in self.flats.columns:
                if self.flats[col].dtype == 'object':
                    self.flats[col] = self.flats[col].astype(float, errors='ignore')
            self.project_metrics = calculate_metrics(self.flats)
        else: print('Empty data')

    def visualisation(self, prj_title=None, prj_id=None):
        if not prj_title and not prj_id:
            print('Please enter Project title or Project_id!')
            return self
        elif not prj_id:
            try:
                prj_id = self.projects[self.projects['prj_title'] == prj_title]['proj_id'].values[0]
            except IndexError:
                print('Project not found')
                return self
        try:
            prj = [p for p in self.project_metrics if p['prj_id'] == prj_id][0]
        except IndexError:
            print('Project no found')
            return self
        visualisation_project(prj)





