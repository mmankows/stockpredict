from csv import reader, DictReader
import logging

from stockpredict.network import SessionData


logger = logging.getLogger()


class NetworkTrainer(object):
    """
    Trains neural network with data
    """

    def __init__(self):
        self._network = None

    def set_network(self, network):
        self._network = network

    def train_from_csv(self, csv_path, col_dict=None):
        col_dict = col_dict or {}
        print csv_path
        data_cnt = 0
        error_cnt = 0
        with open(csv_path, 'rb') as csv_file:
            reader = DictReader(csv_file)
            for row in reader:

                try:
                    SessionData(
                        date=row[col_dict['DATE']],
                        price_open=row[col_dict['OPEN']],
                        price_close=row[col_dict['CLOSE']],
                        price_low=row[col_dict['LOW']],
                        price_high=row[col_dict['HIGH']],
                        volume=row[col_dict['VOLUME']]
                    )
#                    logger.debug("Row parsed: {}".format(row))
                    data_cnt += 1

                except KeyError as e:
                    raise KeyError("CSV header doesn't match provided column mapping!: {}".format(e))

                except TypeError:
                    logger.error("SKIPPED RECORD! - bad data type")
                    error_cnt += 1

        logger.info("Parsed {} data rows".format(data_cnt))
        if error_cnt:
            logger.warn("{} rows skipped due to errors".format(error_cnt))