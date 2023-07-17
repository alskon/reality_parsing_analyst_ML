import psycopg2
from psycopg2.extras import execute_batch
import os
from Databases.connector import Conn_DB
from .config import PARAMS


class RealityDatabase(Conn_DB):
    def __init__(self, host=PARAMS['host'], user=PARAMS['user'], password=PARAMS['password'], db=PARAMS['db']):
        super().__init__(host, user, password, db)
        self.projects = None
        self.flats = None
        self.cities = self.get_city()
        self.developers = self.get_developers()

    def get_city(self):
        cities = []
        request = '''
        select city_code, lower(short_name) from city;
        '''
        if self.cursor:
            self.cursor.execute("select 1 from information_schema.tables where table_name='city'")
            exist_table = self.cursor.fetchone()
            if exist_table:
                self.cursor.execute(request)
                data = self.cursor.fetchall()
                for city in data:
                    cities.append({
                        'city_code': city[0],
                        'short_name': city[1],
                    })
        return cities

    def get_developers(self):
        developers = []
        request = '''
        select id, lower(name) from developer;
        '''
        if self.cursor:
            self.cursor.execute("select 1 from information_schema.tables where table_name='developer'")
            exist_table = self.cursor.fetchone()
            if exist_table:
                self.cursor.execute(request)
                data = self.cursor.fetchall()
                for dev in data:
                    developers.append({
                        'id': dev[0],
                        'name': dev[1],
                    })
        return developers

    def get_all_proj_from_db(self):
        request = f'''
            select * from project;
        '''
        self.cursor.execute(request)
        self.projects = self.cursor.fetchall()

    def get_all_flats_from_db(self):
        request = f'''
            select * from flat;
        '''
        self.cursor.execute(request)
        self.flats = self.cursor.fetchall()

    def get_proj_from_db(self, title_proj: str):
        request = f'''
            select * from project
            where title = %s;
        '''
        self.cursor.execute(request, (title_proj, ))
        elements = self.cursor.fetchall()
        return elements

    def get_flats_from_db(self, title_proj: str):
        request = f'''
            select * from flat f
            join project p
            on f.project_id = p.id
            join city c
            on c.city_code = p.city_code
            where title = %s;
        '''
        self.cursor.execute(request, (title_proj, ))
        elements = self.cursor.fetchall()
        return elements

    def fetch_proj_to_db(self, projects: dict):
        if not self.cursor:
            print('No database connected!')
        else:
            if not self.cities:
                self.cities = self.get_city()
            if not self.developers:
                self.developers = self.get_developers()
            proj_list = []
            metro_list = []
            for pr in projects:
                dev_id = [el['id'] for el in self.developers if el['name'] == pr['developer'].lower()][0]
                city_code = [el['city_code'] for el in self.cities if el['short_name'] == pr['settlement'].lower()][0]
                if pr['metro_title']:
                    metro_exist = True
                else:
                    metro_exist = False
                obj_proj = tuple([dev_id, pr['id'], city_code, pr['title'], metro_exist, pr['latitude'],
                                  pr['longitude'], pr['min_price'], pr['count'], pr['distance_to_mkad']])
                proj_list.append(obj_proj)

                if metro_exist:
                    obj_metro = tuple(
                        [pr['title'], pr['metro_title'], pr['metro_time_on_foot'], pr['metro_time_on_car']])
                    metro_list.append(obj_metro)
            request = '''
                insert into project(
                developer_id, proj_developer_id, city_code, title, metro_exist, latitude, longitude,
                min_price, count_obj, distance_to_mkad_km) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                on conflict (title) 
                do 
                update set developer_id = excluded.developer_id, proj_developer_id = excluded.proj_developer_id, 
                city_code = excluded.city_code, metro_exist = excluded.metro_exist, 
                latitude = excluded.latitude, longitude = excluded.longitude, min_price = excluded.min_price, 
                count_obj = excluded.count_obj, distance_to_mkad_km = excluded.distance_to_mkad_km;
            '''
            execute_batch(self.cursor, request, proj_list)
            self.connector.commit()
            if metro_exist:
                request_metro = f'''
                    insert into metro(
                    project_title, metro_title, metro_min_on_foot, metro_min_on_car) 
                    values (%s,%s,%s,%s)
                    on conflict (project_title) 
                    do 
                    update set metro_title = excluded.metro_title, metro_min_on_foot = excluded.metro_min_on_foot, 
                    metro_min_on_car = excluded.metro_min_on_car;
                '''
            execute_batch(self.cursor, request_metro, metro_list)
            self.connector.commit()

    def fetch_flats_to_db(self, flats: dict):
        if not self.cursor:
            print('No database connected!')
        else:
            projects_fl = tuple(set([pr['title_project'] for pr in flats]))
            print(projects_fl)
            request_proj_id = f'''
            select id, title from project
            where title in %s;
            '''
            try:
                self.cursor.execute(request_proj_id, (projects_fl, ))
                proj_db = self.cursor.fetchall()
            except psycopg2.Error as err:
                print(err)
            if not proj_db:
                print('No project found!')
            else:
                flat_list = []
                proj_db = [{el_db[1]: el_db[0]} for el_db in proj_db]
                for flat in flats:
                    if flat['price'] == 0 or flat['area'] == 0 or flat['price'] is None or \
                            flat['area'] is None or flat['rooms'] == 0 or flat['rooms'] is None or \
                            flat['id'] == 0 or flat['id'] is None:
                        continue
                    proj_id = [pr[flat['title_project']] for pr in proj_db if flat['title_project'] in pr.keys()][0]
                    obj_flat = tuple([proj_id, flat['id'], flat['rooms'], flat['area'], flat['floor'], flat['max_floor'],
                                     flat['whitebox'], flat['furnishing'], flat['balcony'], flat['business'],
                                     flat['design'], flat['price'] ])
                    flat_list.append(obj_flat)
                request = '''
                insert into flat(
                    project_id, flat_developer_id, rooms, area, floor, max_floor, whitebox, furnishing,
                    balcony, business, design, price) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    on conflict on constraint flat_proj_id   
                    do 
                    update set rooms = excluded.rooms, area = excluded.area, floor = excluded.floor,
                    max_floor = excluded.max_floor, whitebox = excluded.whitebox, furnishing = excluded.furnishing,
                    balcony = excluded.balcony, design = excluded.design, price = excluded.price;
                                   
                '''
                try:
                    execute_batch(self.cursor, request, flat_list)
                    self.connector.commit()
                except psycopg2.Error as err:
                    print(err)
