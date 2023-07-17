import pandas as pd
import requests
import datetime
from bs4 import BeautifulSoup as BS


class ParseData:
    def __init__(self, api, **kwargs):
        self.api = api
        self.projects = []
        self.list_projects = []
        self.flats = []
        self.flats_url = None
        self.projects_url = None
        self.headers = {"user-agent": "Mozilla/5.0"}
        self.project_params = None
        self.flats_params = None
        self.flat_count = 0

        if 'url' in kwargs.keys():
            if 'flats' in kwargs['url'].keys():
                self.flats_url = kwargs['url']['flats']
            if 'projects' in kwargs['url'].keys():
                self.projects_url = kwargs['url']['projects']
                self.main_url = '/'.join(kwargs['url']['projects'].split('/')[:3])
        if 'headers' in kwargs.keys():
            self.headers = kwargs['headers']
        if 'params' in kwargs.keys():
            if 'projects' in kwargs['params'].keys():
                self.project_params = kwargs['params']['projects']
            if 'flats' in kwargs['params'].keys():
                self.flats_params = kwargs['params']['flats']

    def parse(self, type_f_p):
        if type_f_p == 'project' and self.projects_url:
            url = self.projects_url
            params = self.project_params
        elif type_f_p == 'flat' and self.flats_url:
            url = self.flats_url
            params = self.flats_params
        else:
            return
        data = requests.request('get', url, headers=self.headers, params=params)
        if self.api:
            data = data.json()
        else:
            data = BS(data.text, features='html.parser')
        return data

    @staticmethod
    def save_to_csv(obj_list: list, path: str, file_name: str):
        date = str(datetime.datetime.now()).split('.')[0].replace('-', '').replace(' ', '').replace(':', '')
        obj_df = pd.DataFrame(obj_list)
        obj_df.to_csv(f'{path}/{file_name}_{date}.csv', index=False)


