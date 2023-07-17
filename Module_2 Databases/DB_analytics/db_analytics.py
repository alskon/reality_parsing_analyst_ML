from psycopg2.extras import execute_batch
import pandas as pd
from Databases.connector import Conn_DB
from .config import PARAMS_INDB,PARAMS_OUTDB


def load_data_from_realityDB(host=PARAMS_INDB['host'], user=PARAMS_INDB['user'],
                                password=PARAMS_INDB['password'], db=PARAMS_INDB['db']):
    connectDB = Conn_DB(host, user, password, db)
    if not connectDB.cursor:
        print('No connection!')
    else:
        request_cur_price = '''
            select fp.id_flat, fp.price, f.project_id, f.area, f.whitebox, f.rooms 
            from flat_price fp
            join flat f on fp.id_flat = f.id
            where dtime::date = current_timestamp::date;
        '''
        connectDB.cursor.execute(request_cur_price)
        cur_price_el = connectDB.cursor.fetchall()
        if not cur_price_el:
            print ('No today data!')
            projects = []
            flats = []
        else:
            current_id_flat = tuple([i[0] for i in cur_price_el])
            current_id_prj = tuple([i[2] for i in cur_price_el])
            request_prev_price = '''
                with prev_price as (
                select row_number() over(partition by id_flat order by dtime desc) num,
                dtime, id_flat, price from flat_price) 
                select id_flat, dtime, price
                from prev_price
                where num = 2 and id_flat in %s;        
            '''
            connectDB.cursor.execute(request_prev_price, (current_id_flat,))
            prev_price_el = connectDB.cursor.fetchall()
            prev_price = [{'flat_id': el[0], 'prev_price': el[2]} for el in prev_price_el]
            request_last_price = '''
                with first_price as (
                select row_number() over(partition by id_flat order by dtime) num,
                dtime, id_flat, price from flat_price) 
                select id_flat, dtime, price
                from first_price
                where num = 1 and id_flat in %s;
            '''
            connectDB.cursor.execute(request_last_price, (current_id_flat,))
            first_price_el = connectDB.cursor.fetchall()
            first_price = [{'flat_id': el[0], 'first_date': el[1], 'first_price': el[2]} for el in first_price_el]
            request_projects = '''
                select prj.id, prj.title, dev.name, city.name, prj.latitude, prj.longitude, city.country
                from project prj join developer dev on prj.developer_id = dev.id
                join city on prj.city_code = city.city_code
                where prj.id in %s;
            '''
            connectDB.cursor.execute(request_projects, (current_id_prj,))
            projects_el = connectDB.cursor.fetchall()
            projects = [
                {'prj_id': el[0],
                'prj_title': el[1],
                'developer_title': el[2],
                'city_name': el[3],
                'latitude': el[4],
                'longitude': el[5],
                'country':el[6]} for el in projects_el]
            flats = [
                {'flat_id': el[0],
                 'today_price': el[1],
                 'prj_id': el[2],
                 'area': el[3],
                 'whitebox': el[4],
                 'rooms': el[5]} for el in cur_price_el]
            flats = pd.DataFrame(flats).merge(pd.DataFrame(prev_price),
                                    how='outer').fillna('0').to_dict('records')
            flats = pd.DataFrame(flats).merge(pd.DataFrame(first_price),
                                    how='outer').fillna('0').to_dict('records')
        connectDB.cursor.close()
        connectDB.connector.close()
        return (projects, flats)

class AnalyticsDB(Conn_DB):
    def __init__(self, host=PARAMS_OUTDB['host'], user=PARAMS_OUTDB['user'],
                 password=PARAMS_OUTDB['password'], db=PARAMS_OUTDB['db']):
        super().__init__(host, user, password, db)
        self.projects_from_reality = []
        self.flats_from_reality = []
        self.projects_from_analytics = []
        self.flats_from_analytics = []

    def load_data(self):
        self.projects_from_reality, self.flats_from_reality = load_data_from_realityDB()

    def fetch_projects_to_DB(self):
        if not self.cursor:
            print('No connection')
        else:
            req_truncate = '''
                truncate table project;
            '''
            self.cursor.execute(req_truncate)
            self.connector.commit()
            req_insert = '''
                insert into project 
                values(%s, %s, %s, %s, %s, %s, %s);
            '''
            prj_data = [(el['prj_id'], el['prj_title'], el['developer_title'], el['city_name'],
                         el['country'], el['latitude'], el['longitude'], ) for el in self.projects_from_reality]
            execute_batch(self.cursor, req_insert, prj_data)
            self.connector.commit()

    def fetch_flats_to_DB(self):
        if not self.cursor:
            print('No connection')
        else:
            req_truncate = '''
                truncate table flat;
            '''
            self.cursor.execute(req_truncate)
            self.connector.commit()
            req_insert = '''
                insert into flat
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            '''
            flat_data = [(el['flat_id'], el['prj_id'], el['area'], el['rooms'], el['whitebox'], el['today_price'],
                          el['prev_price'], el['first_price'], el['first_date']) for el in self.flats_from_reality]
            execute_batch(self.cursor, req_insert, flat_data)
            self.connector.commit()

    def get_data(self, project_title=None):
        if not self.cursor:
            print('No connection DB')
        else:
            if project_title:
                req_prj = '''
                    select * from project
                    where prj_title = %s;
                '''
                req_flats = '''
                    select * from flat
                    where prj_id = (select id from project where prj_title = %s); 
                '''
            else:
                req_prj = '''
                    select * from project;
                '''
                req_flats = '''
                    select * from flat;
                '''
            self.cursor.execute(req_prj, (project_title, ))
            proj =  self.cursor.fetchall()
            self.cursor.execute(req_flats, (project_title, ))
            flats = self.cursor.fetchall()

            self.projects_from_analytics = [{
                'proj_id': el[0],
                'prj_title': el[1],
                'developer_title': el[2],
                'city_name': el[3],
                'country': el[4],
                'latitude': el[5],
                'longitude': el[6]
            } for el in proj]

            self.flats_from_analytics = [{
                'flat_id': el[0],
                'prj_id': el[1],
                'area': el[2],
                'rooms': el[3],
                'whitebox': el[4],
                'today_price': el[5],
                'prev_price': el[6],
                'first_price': el[7],
                'date_first_price': el[8]
            } for el in flats]








