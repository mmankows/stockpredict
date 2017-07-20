from csv import DictReader

from stockpredict.network import SessionData


class NetworkTrainer(object):

    def __init__(self):
        pass

    def load_session_data(self, csv_path, col_dict):
        with open(csv_path, 'r') as csv_file:
            reader = DictReader(csv_path)
            for row in reader:
                yield SessionData(
                    date=row[col_dict['DATE']],
                    price_open=row[col_dict['PRICE_OPEN']],
                    price_close=row[col_dict['PRICE_CLOSE']],
                    price_low=row[col_dict['PRICE_LOW']],
                    price_high=row[col_dict['PRICE_HIGH']],
                    volume=row[col_dict['VOLUME']]
                )