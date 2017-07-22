from csv import reader, DictReader
import logging

from dateutil.parser import parse

from stockpredict.network import SessionData


logger = logging.getLogger()


class TrainingManager(object):
    """
    Trains neural network with data
    """

    def __init__(self):
        self._network = None

    def set_network(self, network):
        self._network = network

    def splited_datasets(self, sessions_data):
        """
        Generates training sets from training data
        :return: 
        """
        network_params = self._network.params
        max_data_samples = len(sessions_data) - network_params.sessions_tracked - network_params.sessions_predicted
        # max_training_cycles = 5

        # TODO - only freshest data now
        for cycle_num in range(0, 30):#)max_data_samples+1):
            last_tracked_idx = cycle_num+network_params.sessions_tracked
            # normalize dataset parameters
            training_vector = sessions_data[cycle_num: last_tracked_idx + network_params.sessions_predicted]
            training_vector = self.normalize_session_data(training_vector)
            yield (
                training_vector[:network_params.sessions_tracked],
                training_vector[network_params.sessions_tracked:]
            )

    def normalize_session_data(self, session_data):
        normalized_sd = []

        # calculate volume mean
        volume_mean = float(sum(sd.volume for sd in session_data)) / len(session_data)

        logger.debug('Total volume: {}'.format(sum(sd.volume for sd in session_data)))
        logger.debug('Volume mean: {:.2f}'.format(volume_mean))

        def cap(price):
            if price > 1:
                logger.warn('Capped - max')
                return 1
            elif price < -1:
                logger.warn('Capped - min')
                return -1
            else:
                return price

        prev_sd = session_data[0]
        for sd in session_data:
            normalized_sd.append(SessionData(
                timestamp=sd.timestamp,
                volume=cap((float(sd.volume) - float(volume_mean)) / float(volume_mean)),
                **{price: cap((getattr(sd, price) - getattr(prev_sd, price)) / getattr(prev_sd, price))
                   for price in ['price_open', 'price_close', 'price_high', 'price_low']
                }
            ))

        for nsd in normalized_sd:
            logger.debug("SD: TS{}\tVM{:.6f}\tPO{:.6f}\tPC{:.6f}\tPH{:.6f}\tPL{:.6f}".format(nsd.timestamp, nsd.volume, *[getattr(nsd, price) for price in ['price_open', 'price_close', 'price_high', 'price_low']]))

        return normalized_sd

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
                        timestamp=parse(row[col_dict['DATE']]),
                        price_open=float(row[col_dict['OPEN']]),
                        price_close=float(row[col_dict['CLOSE']]),
                        price_low=float(row[col_dict['LOW']]),
                        price_high=float(row[col_dict['HIGH']]),
                        volume=int(row[col_dict['VOLUME']])
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

        self._network.train(self.splited_datasets(all_sessions_data))
        # ds = None
        # for ds in self.splited_datasets(all_sessions_data):
        #
        #     self._network.predict(
        #         ds[0], ds[1]
        #     )
        # self._network.display()

