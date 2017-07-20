from pybrain.structure import FeedForwardNetwork, SigmoidLayer
from dateutil.parser import parse


class StockNeuralNetwork(object):

    def __init__(self, previous=365, future=10):
        self._net = FeedForwardNetwork()

        # Initialize input modules for each session in past - volume, open,
        for _ in range(previous):
            self._net.addInputModule(SigmoidLayer(3))

        for _ in range(3):
            self._net.addModule(SigmoidLayer(3))

        # Output per each future session - we're interested in volume & close
        for _ in range(future):
            self._net.addOutputModule(SigmoidLayer(2))

    def train(self, sessions_data):
        """
        
        :param sessions_data: List sessions 
        :return: 
        """
        for session_data in sessions_data:
            pass


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