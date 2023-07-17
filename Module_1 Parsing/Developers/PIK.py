import datetime
from Parsing.main_parser import ParseData
import json


# --- PIK ---
class ParsePik(ParseData):
    def __init__(self, api=False, **kwargs):
        super().__init__(api, **kwargs)

    def get_projects(self):
        raw_data = self.parse('project')
        raw_data = raw_data.find('script', id='__NEXT_DATA__').text
        self.flat_count = json.loads(raw_data)['props']['pageProps']['initialState'] \
            ['searchService']['filteredProjects']['data']['count']
        json_project_list = json.loads(raw_data)['props']['pageProps']['initialState'] \
            ['searchService']['filteredProjects']['data']['projects']
        for obj in json_project_list:
            if obj['locations']['parent']['name'] == 'Москва':
                settle = 'msk'
            elif obj['locations']['parent']['name'] == 'Московская область':
                settle = 'msko'
            else:
                settle = obj['locations']['parent']['name']
            short_obj = {
                'developer': 'PIK',
                'settlement': settle,
                'id': obj['id'],
                'title': obj['name'],
                'metro_title': obj['metro'],
                'metro_time_on_foot': obj['timeOnFoot'],
                'metro_time_on_car': obj['timeOnTransport'],
                'distance_to_mkad': None,  # todo calculate distance
                'latitude': obj['latitude'],
                'longitude': obj['longitude'],
                'min_price': obj['priceMin'],
                'count': obj['count'],
                'ref': '/'+ obj['url']
            }
            self.projects.append(short_obj)
        self.list_projects = [project['title'] for project in self.projects]

    def get_flats(self):
        raw_data = self.parse('flat')
        raw_data = raw_data.find('script', id='__NEXT_DATA__').text
        last_page = json.loads(raw_data)['props']['pageProps']['initialState'] \
            ['searchService']['filteredProjects']['data']['lastPage']
        for i in range(2, last_page + 1):
            json_flat_list = json.loads(raw_data)['props']['pageProps']['initialState'] \
            ['searchService']['filteredProjects']['data']['flats']

            for flat in json_flat_list:

                if flat['rooms'] == -1:
                    rooms = 0
                else: rooms = flat['rooms']

                flat_project_id = None
                flat_project_title = None
                if self.projects:
                    if flat['block']['id'] in [prj['id'] for prj in self.projects]:
                        flat_project_id = flat['block']['id']
                        flat_project_title = flat['block']['name']
                else: print('Recommend firstly launch get_projects')

                if not flat['bulk']['settlementDate']:
                    date = 0
                else:
                    split_date = flat['bulk']['settlementDate'].split('-')
                    split_date = [int(ch) for ch in split_date]
                    date = (split_date[0] - datetime.datetime.now().year) * 12 + \
                           (split_date[1] - datetime.datetime.now().month)

                if flat['floor'] == 1:
                    balcony = False
                else:
                    balcony = True

                short_flat = {
                    'id': flat['id'],
                    'title_project': flat_project_title,
                    'title_id': flat_project_id,
                    'months_to_end': date,
                    'rooms': rooms,
                    'area': flat['area'],
                    'price': flat['price'],
                    'floor': flat['floor'],
                    'max_floor': None,
                    'design': False,
                    'whitebox': flat['whiteBox'],
                    'furnishing': flat['furniture'],
                    'business': False,
                    'balcony': balcony,
                }
                self.flats.append(short_flat)
            self.flats_params =  {
                'flatPage': i,
                'allFlats': 1
            }
            raw_data = self.parse('flat')
            raw_data = raw_data.find('script', id='__NEXT_DATA__').text
            print(len(self.flats))