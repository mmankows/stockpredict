import cli.app
import logging

from stockpredict.training import TrainingManager
from stockpredict.network import StockNeuralNetwork

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


@cli.app.CommandLineApp
def train(app):
    """
    Trains a network with a CSV file data
    :param app: 
    :return: 
    """
    # # Set log level
    # assert app.params.log_level in ['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
    # logger.debug("DEBUG")
    # logger.info("INFO")
    # logger.warn("WARN")
    # logger.error("ERROR")
    # logger.critical("CRITICAL")
    # logging.basicConfig(level=getattr(logging, app.params.log_level))

    # Choose Trainer
    trainer = TrainingManager()

    # Choose network for or create new one
    logger.info("Setting up network.")
    net = StockNeuralNetwork(previous=int(app.params.previous) or 50, future=int(app.params.future) or 3)
    trainer.set_network(net)

    # Run training
    logger.info("Starting training from {} file.".format(app.params.input_csv))
    trainer.train_from_csv(csv_path=app.params.input_csv, col_dict={
        'DATE': 'Data',
        'OPEN': 'Otwarcie',
        'CLOSE': 'Zamkniecie',
        'LOW': 'Najnizszy',
        'HIGH': 'Najwyzszy',
        'VOLUME': 'Wolumen'
    })
    # Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen


train.add_param("-c", "--input-csv", help="Input csv file with stock data", required=True, action="store")
train.add_param("-p", "--previous", help="Number of previous sessions (analyzed)", required=False, action="store")
train.add_param("-f", "--future", help="Number of future sessions (predicted)", required=False, action="store")
train.add_param("-l", "--log-level", help="Log level (default is WARN)", required=False, default='WARN', action="store")

if __name__ == "__main__":
    train.run()
