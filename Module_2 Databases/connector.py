import psycopg2
import os

class Conn_DB:
    def __init__(self, host, user, password, db):
        try:
            self.connector = psycopg2.connect(
                host=host,
                database=db,
                user=user,
                password=password,
            )
            print(f'Connect to database {db}!')
        except psycopg2.OperationalError as err:
            print(f'Cannot connect to database, error: {err}')
            self.connector = None

        if self.connector:
            self.cursor = self.connector.cursor()
        else: self.cursor = None

    def __del__(self):
        self.cursor.close()
        self.connector.close()

    def run_script(self, script):
        if self.cursor:
            try:
                self.cursor.execute(script)
                self.connector.commit()
                print('Script done')
            except psycopg2.Error as err:
                print(err)
                self.connector.rollback()
        else: print('No cursor')

    def oper_tables(self, command, scripts_path):
        scripts = []
        if command in scripts_path.keys():
            path = scripts_path[command]
            for dirname, _, filenames in os.walk(path):
                print(dirname, filenames)
                for filename in filenames:
                    scripts.append(os.path.join(dirname, filename))
        if self.cursor:
            for script in scripts:
                with open(script, encoding="utf8") as scr:
                    f = scr.read()
                    print(f)
                try:
                    self.run_script(f)
                except psycopg2.Error as err:
                    print(f'Cannot do operation, error: {err}')

