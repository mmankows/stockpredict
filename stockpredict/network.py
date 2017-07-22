import logging

from datetime import datetime
from pybrain.structure import FeedForwardNetwork, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from dateutil.parser import parse


logger = logging.getLogger()

INPUT_PARAMS_COUNT = 5


class StockNeuralNetwork(object):
    """
    Neural network for stock data
    """

    def __init__(self, previous=365, future=10):
        self._net = FeedForwardNetwork()
        self.params = NetworkParams(sessions_tracked=previous, sessions_predicted=future)

        # --- Initialize network layers
        # 1. Initialize input modules for each session in past -
        input_modules = [SigmoidLayer(INPUT_PARAMS_COUNT) for _ in range(previous)]
        for in_module in input_modules:
            self._net.addInputModule(in_module)

        # 2. Hidden layers
        hidden_layers = [SigmoidLayer(30) for _ in range(3)]
        for h_layer in hidden_layers:
            self._net.addModule(h_layer)

        # 3. Output per each future session - we're interested in volume & close
        output_modules = [SigmoidLayer(INPUT_PARAMS_COUNT) for _ in range(future)]
        for out_module in output_modules:
            self._net.addOutputModule(out_module)

        # --- Define connections
        # 1. All input module to bottom hidden layers
        for in_module in input_modules:
            self._net.addConnection(
                FullConnection(in_module, hidden_layers[0])
            )

        # 2. each hidden layer connected
        for idx, h_layer in enumerate(hidden_layers):
            if idx > 0:
                self._net.addConnection(
                    FullConnection(h_layer, hidden_layers[idx-1])
                )

        # 3. top hidden layer to all output modules
        for out_module in output_modules:
            self._net.addConnection(
                FullConnection(out_module, hidden_layers[-1])
            )

        # --- magic
        self._net.sortModules()

    def train(self, input_output_pairs):
        """
        
        :param input_output_pairs: 
        :return: 
        """

        # Construct dataset
        dataset = SupervisedDataSet(INPUT_PARAMS_COUNT * self.params.sessions_tracked, INPUT_PARAMS_COUNT * self.params.sessions_predicted)
        for io_pair in input_output_pairs:
            print io_pair
            input_sessions = io_pair[0]
            output_sessions = io_pair[1]

            dataset.appendLinked(
                [sd for session in input_sessions for sd in session.to_vector()],
                [sd for session in output_sessions for sd in session.to_vector()]
            )

        # training
        trainer = BackpropTrainer(self._net, dataset=dataset)
        for i in range(1000):
            err = trainer.train()
            print("ERR: %s" % err)
        # ret = trainer.trainUntilConvergence(dataset, verbose=True)
        # print ret

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
    def __init__(self, timestamp=None, volume=None, price_open=None, price_close=None, price_high=None, price_low=None):
        self.timestamp = parse(timestamp) if type(timestamp) in (str, unicode) else timestamp
        self.volume = volume
        self.price_open = price_open
        self.price_close = price_close
        self.price_high = price_high
        self.price_low = price_low

    def to_vector(self):
        """
        Convert session data to linear vector - all non - nullable dimensions
        :return: 
        """
        return [getattr(self, name) for name in
                ['volume', 'price_open', 'price_close', 'price_high', 'price_low']
                if getattr(self, name) is not None
        ]