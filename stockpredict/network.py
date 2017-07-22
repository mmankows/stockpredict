import logging

from datetime import datetime
from pybrain.structure import FeedForwardNetwork, SigmoidLayer, FullConnection, TanhLayer, LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from dateutil.parser import parse


logger = logging.getLogger()

INPUT_PARAMS_COUNT = 5
HIDDEN_LAYERS_COUNT = 3


class StockNeuralNetwork(object):
    """
    Neural network for stock data
    """

    def __init__(self, previous, future):
        self._net = FeedForwardNetwork()
        self.params = NetworkParams(sessions_tracked=previous, sessions_predicted=future)

        # --- Initialize network layers
        # 1. Initialize input modules for each session in past -
        input_modules = [TanhLayer(INPUT_PARAMS_COUNT) for _ in range(previous)]
        for in_module in input_modules:
            self._net.addInputModule(in_module)

        # 2. Hidden layers
        hidden_neurons_count = [previous, previous, previous]
        hidden_layers = [TanhLayer(hidden_neurons_count.pop(0)) for _ in range(HIDDEN_LAYERS_COUNT)]
        for h_layer in hidden_layers:
            self._net.addModule(h_layer)

        # 3. Output per each future session - we're interested in volume & close
        output_modules = [TanhLayer(INPUT_PARAMS_COUNT) for _ in range(future)]
        for out_module in output_modules:
            self._net.addOutputModule(out_module)

        # --- Define connections - order matters - bottom to top!!!
        # 1. All input module to bottom hidden layers
        for in_module in input_modules:
            self._net.addConnection(
                FullConnection(in_module, hidden_layers[0])
            )

        # 2. each hidden layer connected
        for idx, h_layer in enumerate(hidden_layers):
            if idx > 0:
                self._net.addConnection(
                    FullConnection(hidden_layers[idx-1], h_layer)
                )

        # 3. top hidden layer to all output modules
        for out_module in output_modules:
            self._net.addConnection(
                FullConnection(hidden_layers[-1], out_module)
            )

        # --- magic
        self._net.sortModules()
        self._net.randomize()
        # self._net.convertToFastNetwork()

    def train(self, input_output_pairs):
        """
        
        :param input_output_pairs: 
        :return: 
        """
        # Construct dataset
        dataset = SupervisedDataSet(
            INPUT_PARAMS_COUNT * self.params.sessions_tracked,
            INPUT_PARAMS_COUNT * self.params.sessions_predicted
        )
        for io_pair in input_output_pairs:
            input_sessions = io_pair[0]
            output_sessions = io_pair[1]

            #dataset.appendLinked(
            dataset.addSample(
                [sd for session in input_sessions for sd in session.to_vector()],
                [sd for session in output_sessions for sd in session.to_vector()]
            )
        logger.info("Training with {} samples...".format(len(dataset)))
        # training
        trainer = BackpropTrainer(self._net, dataset=dataset)
        # print datetime.now()
        # trainer.trainUntilConvergence(validationProportion=0.75)
        # print datetime.now()
        for i in range(10000):
            err = trainer.train()
            print("ERR: %s" % err)

    def predict(self, input_sessions, output_sessions=None):
        input_sessions_vector = [sd for session in input_sessions for sd in session.to_vector()]
        # print ("Test activation for %s" % input_sessions_vector)
        print self._net.activate(input_sessions_vector)
        if output_sessions:
             print ("should be: %s" %[sd for session in output_sessions for sd in session.to_vector()])

    def display(self):
        net = self._net
        repr_str = ''
        for mod in net.modules:
            repr_str += "Module: %s\n" % mod.name
            if mod.paramdim > 0:
                repr_str += "--parameters: %s\n" % mod.params
            for conn in net.connections[mod]:
                repr_str += "-connection to %s\n" % conn.outmod.name
                if conn.paramdim > 0:
                    repr_str += "- parameters %s\n" % conn.params
            if hasattr(net, "recurrentConns"):
                print("Recurrent connections")
                for conn in net.recurrentConns:
                    print("-", conn.inmod.name, " to", conn.outmod.name)
                    if conn.paramdim > 0:
                        print("- parameters", conn.params)
        return repr_str

    def save_as_file(self, path):
        pass

    @classmethod
    def load_from_file(cls, path):
        return StockNeuralNetwork()


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