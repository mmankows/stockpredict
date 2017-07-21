import logging
from pybrain.structure import FeedForwardNetwork, SigmoidLayer
from dateutil.parser import parse


logger = logging.getLogger()


class StockNeuralNetwork(object):
    """
    Neural network for stock data
    """

    def __init__(self, previous=365, future=10):
        self._net = FeedForwardNetwork()
        self.params = NetworkParams(sessions_tracked=previous, sessions_predicted=future)

        # Initialize input modules for each session in past - volume, open,
        for _ in range(previous):
            self._net.addInputModule(SigmoidLayer(3))

        for _ in range(3):
            self._net.addModule(SigmoidLayer(3))

        # Output per each future session - we're interested in volume & close
        for _ in range(future):
            self._net.addOutputModule(SigmoidLayer(2))

    def train(self, input_sessions, output_sessions):
        """
        :param sessions_data: List sessions 
        :return: 
        """
        pass


class NetworkParams(object):
    """
    Storage for network parameters
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class SessionData(object):
    """
    SessionData - represent
    """
    def __init__(self, date, volume, price_open, price_close, price_high, price_low):
        self.date = parse(date).date()
        self.volume = int(volume)
        self.price_open = float(price_open)
        self.price_close = float(price_close)
        self.price_high = float(price_high)
        self.pric_low = float(price_low)