import datetime
from Parsing.main_parser import ParseData


# --- A101 ---
class ParseA101(ParseData):
    def __init__(self, api=True, **kwargs):
        super().__init__(api, **kwargs)

    def get_projects(self):
        data = self.parse('project')
        for obj in data['results']:
            if obj['count'] > 0:
                try:
                    distance_to_mkad = float(obj['distance_to_mkad'].replace(',','.'))
                except ValueError:
                    distance_to_mkad = None
                short_obj = {'developer': 'A101',
                             'settlement': obj['settlement_slug'],
                             'id': obj['id'],
                             'title': obj['title'],
                             'metro_title': obj['metro_set'][0]['title'],
                             'metro_time_on_foot': obj['metro_set'][0]['time_on_foot'],
                             'metro_time_on_car': obj['metro_set'][0]['time_on_car'],
                             'distance_to_mkad': distance_to_mkad,
                             'latitude': obj['latitude'],
                             'longitude': obj['longitude'],
                             'min_price': float(obj['min_price'].replace(',','.'))*1000000,
                             'count': obj['count'],
                             'ref': obj['url']
                             }
                self.projects.append(short_obj)
            self.list_projects = [project['title'] for project in self.projects]

    def get_flats(self):
        self.flat_count = self.parse('flat')['count']
        self.flats_params = {'limit': self.flat_count}
        list_flats = self.parse('flat')['results']

        for flat in list_flats:
            if flat['stage'] == 'С ключами':
                date = 0
            else:
                period = flat['stage'].split(' ')
                year_span = int(period[2]) - datetime.datetime.now().year
                if period[0] in {'I', 'II', 'III'}:
                    month_span = len(period[0]) * 3 - datetime.datetime.now().month
                else:
                    month_span = 12 - datetime.datetime.now().month
                date = year_span * 12 + month_span

            try:
                floor = int(flat['floor'])
            except:
                floor = int(flat['floor'].split('-')[0])

            if flat['room_name'] == 'Студия':
                rooms = 0
            else: rooms = flat['room']
            flat_project_id = None
            if self.projects:
                for prj in self.projects:
                    if prj['title'] == flat['complex']:
                        flat_project_id = prj['id']
            else: print('Recommend firstly launch get_projects')

            short_flat = {
                'id': flat['id'],
                'title_project': flat['complex'],
                'title_id': flat_project_id,
                'months_to_end': date,
                'rooms': rooms,
                'area': flat['area'],
                'price': flat['actual_price'],
                'floor': floor,
                'max_floor': int(flat['max_floor']),
                'design': bool(flat['design']),
                'whitebox': bool(flat['whitebox']),
                'furnishing': bool(flat['furnishing']),
                'business': bool(flat['business']),
                'balcony': bool(flat['balcony'])
            }
            self.flats.append(short_flat)


