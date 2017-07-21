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

    def train_from_sessions_data(self, sessions_data):
        sessions_data = self.normalize_session_data(sessions_data)
        network_params = self._network.params
        max_training_cycles = len(sessions_data) - network_params.sessions_tracked - network_params.sessions_predicted

        for cycle_num in range(0, max_training_cycles+1):
            logger.debug("Training cycle {}. Offset: T({}-{}), P({}-{})".format(
                cycle_num,
                cycle_num,
                cycle_num+network_params.sessions_tracked+1,
                cycle_num+network_params.sessions_tracked+1,
                cycle_num+network_params.sessions_tracked+network_params.sessions_predicted+1
            ))
            self._network.train(
                input_sessions=sessions_data[cycle_num:cycle_num+network_params.sessions_tracked+1],
                output_sessions=sessions_data[cycle_num+network_params.sessions_tracked+1:cycle_num+network_params.sessions_tracked+network_params.sessions_predicted+1]
            )

    def normalize_session_data(self, session_data):
        for sd in session_data:
            # TODO - normalize
            pass
        return session_data

    def train_from_csv(self, csv_path, col_dict=None):
        col_dict = col_dict or {}

        data_cnt = 0
        error_cnt = 0

        all_sessions_data = []
        with open(csv_path, 'rb') as csv_file:
            reader = DictReader(csv_file)
            for row in reader:

                try:
                    sd = SessionData(
                        date=row[col_dict['DATE']],
                        price_open=row[col_dict['OPEN']],
                        price_close=row[col_dict['CLOSE']],
                        price_low=row[col_dict['LOW']],
                        price_high=row[col_dict['HIGH']],
                        volume=row[col_dict['VOLUME']]
                    )
                    all_sessions_data.append(sd)
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

        self.train_from_sessions_data(all_sessions_data)

