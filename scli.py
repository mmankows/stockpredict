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
    # Choose Trainer
    trainer = TrainingManager()

    # Choose network for or create new one
    logger.info("Setting up network.")
    net = StockNeuralNetwork(previous=5, future=5)
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


train.add_param("-i", "--input-csv", help="Input csv file with stock data", required=True, action="store")

if __name__ == "__main__":
    train.run()
