from .constants import DB_REALITY, DB_ANALYTICS,  MODELS_ANALYSIS
from Databases.connector import Conn_DB
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


def predict_mean_price(series, p=4, d=1, q=4, pred_step=2):
    model = ARIMA(series, order=(p,d,q))
    model.initialize_approximate_diffuse()
    model_fit = model.fit()
    return model_fit.forecast(pred_step)


def calculate_distance(lat1,long1,lat2,long2):
    R = 6371
    lat1, long1, lat2, long2 = float(lat1),float(long1),float(lat2),float(long2)
    deg_rad = np.pi/180
    calc_lat = (lat2 - lat1) * deg_rad
    calc_long = (long2 - long1) * deg_rad
    a = np.sin(calc_lat / 2) * np.sin(calc_lat / 2) + np.cos(lat1*deg_rad) * np.cos(lat2*deg_rad) * \
        np.sin(calc_long / 2) * np.sin(calc_long / 2)
    b = 2 * np.arctan2(a**0.5, (1 - a)**0.5)
    return np.round(R * b, 2)


class Dataset:
    def __init__(self, dataset, target_name, test_size=0.2):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        if target_name in dataset.columns:
            X_train, X_test, y_train, y_test = train_test_split(dataset.drop(labels=[target_name], axis=1),
                                                                dataset[target_name], test_size=test_size)
            self.X_train = X_train.to_numpy()
            self.X_test = X_test.to_numpy()
            self.y_train = y_train.to_numpy().astype('float32')
            self.y_test = y_test.to_numpy().astype('float32')
        else: print('Target not found')

    def get_dataset_train(self):
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
        }

    def get_dataset_test(self):
        return {
            'X_test': self.X_test,
            'y_test': self.y_test
        }


class ModelNN(nn.Module):
    def __init__(self, n_features, bn=True, dp=True, bias=True, bs=64, epochs=40):
        super().__init__()
        self.n_features = n_features
        self.bn = bn
        self.dp = dp
        self.bias = bias
        self.bs = bs
        self.epochs = epochs

        self.linear_1 = self.form_layer(self.n_features, 512)
        self.linear_2 = self.form_layer(512, 1024)
        self.linear_3 = self.form_layer(1024, 512)
        self.linear_4 = nn.Linear(512, 1)

        self.opt = torch.optim.Adam(self.parameters(), lr=0.06)
        self.loss = nn.MSELoss()
        self.history = {
            'loss_train': [],
            'loss_test': [] }

    def form_layer(self, in_n, out_n):
        layers = [nn.Linear(in_n, out_n, bias=self.bias)]
        if self.bn:
            layers.append(nn.BatchNorm1d(out_n))
        layers.append(nn.LeakyReLU(0.1))
        if self.dp:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return self.linear_4(x)

    def train_nn(self, X_train, y_train, X_test, y_test):
        data_train = [(x, y) for x, y in zip(X_train, y_train)]
        data_test = [(x, y) for x, y in zip(X_test, y_test)]
        dataloader_train = DataLoader(data_train, self.bs, drop_last=True)
        dataloader_test = DataLoader(data_test, self.bs, drop_last=True)
        for epoch in range(self.epochs):
            self.train()
            ep_loss = []
            ep_test_loss = []
            for x, y in dataloader_train:
                x = x.type(torch.float32)
                y = y[:, None]
                self.opt.zero_grad()
                train_pred = self(x)
                _loss = self.loss(train_pred, y)
                ep_loss.append(_loss)
                _loss.backward()
                self.opt.step()
            ep_loss = torch.stack(ep_loss).detach().numpy()
            print(f'Epoch: {epoch + 1}')
            print(f'Loss: {np.sum(ep_loss)/ep_loss.shape[0]}')
            self.history['loss_train'].append(sum(ep_loss) / len(ep_loss))
            with torch.no_grad():
                self.eval()
                for x, y in dataloader_test:
                    x = x.type(torch.float32)
                    y = y[:, None]
                    test_pred = self(x)
                    _loss = self.loss(test_pred, y)
                    ep_test_loss.append(_loss)
                ep_test_loss = torch.stack(ep_test_loss).detach().numpy()
                print(f'Test Loss: {np.sum(ep_test_loss)/ep_test_loss.shape[0]}')
                self.history['loss_test'].append(sum(ep_test_loss) / len(ep_test_loss))

    def predict(self, data):
        data = torch.from_numpy(data).type(torch.float32)
        self.eval()
        y_pred = self(data)
        return y_pred.detach().numpy()

class ModelML:
    def __init__(self, model_name='lreg'):
        self.model_name = model_name
        self.model = None
        self.metrics = {
            'r_score': '',
            'RMSE': '',
            'MAE': ''
        }

    def calculate_metrics(self, X_test, y_test):
        prediction = self.model.predict(X_test)
        self.metrics['r_score'] = self.model.score(X_test, y_test)
        self.metrics['MAE'] = mean_absolute_error(y_test, prediction)
        self.metrics['RMSE'] = mean_squared_error(y_test, prediction)**0.5

    def train_model(self, dataset, target_name, std=True):
        ds = Dataset(dataset, target_name)
        X_train, y_train = ds.get_dataset_train()['X_train'], ds.get_dataset_train()['y_train']
        X_test, y_test = ds.get_dataset_test()['X_test'], ds.get_dataset_test()['y_test']
        if std:
            std_scaler = StandardScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.fit_transform(X_test)
        if self.model_name in MODELS_ANALYSIS:
            if self.model_name == 'lreg':
                self.model = LinearRegression().fit(X_train, y_train)
                self.calculate_metrics(X_test, y_test)
            if self.model_name == 'rf':
                self.model = RandomForestRegressor(n_estimators=300).fit(X_train, y_train)
                self.calculate_metrics(X_test, y_test)
            if self.model_name == 'gboost':
                self.model = GradientBoostingRegressor(n_estimators=300).fit(X_train, y_train)
                self.calculate_metrics(X_test, y_test)
            if self.model_name == 'aboost':
                self.model = AdaBoostRegressor(n_estimators=100).fit(X_train, y_train)
                self.calculate_metrics(X_test, y_test)
            if self.model_name == 'nn':
                n_features = X_train.shape[1]
                self.model = ModelNN(n_features)
                X_test_nn, X_test_metrics, y_test_nn, y_test_metrics = train_test_split(X_test, y_test, test_size=0.4)
                self.model.train_nn(X_train, y_train, X_test_nn, y_test_nn)
                prediction = self.model.predict(X_test_metrics)
                self.metrics['r_score'] = r2_score(y_test_metrics, prediction)
                self.metrics['MAE'] = mean_absolute_error(y_test_metrics, prediction)
                self.metrics['RMSE'] = mean_squared_error(y_test_metrics, prediction) ** 0.5
        else: print('No model found')
        print(self.metrics)


class MLAnalysis:
    def __init__(self, host=DB_REALITY['host'], user=DB_REALITY['user'], password=DB_REALITY['password'],
                 db=DB_REALITY['db']):
        self.connectDB = Conn_DB(host, user, password, db)
        self.dataset = None
        self.model = None

    def collect_data(self):
        request = '''
                select f.id, prj.title, city.short_name, f.rooms, 
                f.area, f.floor, prj.latitude, prj.longitude, city.center_lat, city.center_long, f.price
                from flat f join project prj on f.project_id = prj.id
                join city on prj.city_code = city.city_code;
        '''
        if self.connectDB.cursor:
            self.connectDB.cursor.execute(request)
            data = self.connectDB.cursor.fetchall()
            headers = ['flat_id', 'title', 'city', 'rooms', 'area', 'floor',
                       'latitude', 'longitude', 'center_lat', 'center_long', 'price']
            self.dataset = pd.DataFrame(data, columns=headers)
            self.dataset.set_index('flat_id', inplace=True)
            self.dataset['to_center'] = self.dataset.apply(lambda x: calculate_distance(x.latitude, x.longitude,
                                                                                        x.center_lat, x.center_long),
                                                           axis=1)
            self.dataset.drop(labels=['title', 'latitude', 'longitude', 'center_lat', 'center_long'], axis=1, inplace=True)
            self.dataset['area'] = self.dataset['area'].astype(float)
            self.dataset['price'] = self.dataset['price'].astype(float)
        else:
            print('No DB connection')

    def train_dataset(self, model_name='aboost', city='msk'):
        cities = self.dataset['city'].unique()
        if city in cities:
            data = self.dataset.loc[self.dataset['city'] == city].drop(labels=['city'], axis=1)
            self.model = ModelML(model_name)
            self.model.train_model(data, 'price')
        else:
            print('City not found!')

    def predict_data(self, data):
        prediction = self.model.predict(data)
        return prediction


class TimeSeriesModel:
    def __init__(self, host=DB_ANALYTICS['host'], user=DB_ANALYTICS['user'], password=DB_ANALYTICS['password'],
                     db=DB_ANALYTICS['db']):
        self.connectDB = Conn_DB(host, user, password, db)
        self.dataset = None
        self.model = None

    def collect_data(self, prj_title):
        prj_title = prj_title.title()
        request_prj = '''
            select id from project
            where prj_title = %s
            limit 1;
        '''
        request_prj_mean_price = '''
            select * from project_price_mean
            where id_project = %s;
        '''
        if self.connectDB.cursor:
            try:
                self.connectDB.cursor.execute(request_prj, (prj_title, ))
                prj_id = self.connectDB.cursor.fetchall()[0]
            except:
                prj_id = None
            if prj_id:
                self.connectDB.cursor.execute(request_prj_mean_price, prj_id)
                prj_stats_names = [name[0] for name in  self.connectDB.cursor.description]
                prj_stats = self.connectDB.cursor.fetchall()
                prj_stats_pd = pd.DataFrame(prj_stats, columns=prj_stats_names)
                prj_stats_pd['date_price'] = pd.to_datetime(prj_stats_pd['date_price'])
                prj_stats_pd.set_index('date_price', inplace=True)
                prj_stats_pd = prj_stats_pd.astype(float, errors='ignore').sort_values(by='date_price')

                forecast = predict_mean_price(prj_stats_pd['mean_price_m2'])
            else:
                print('Project not exist')
