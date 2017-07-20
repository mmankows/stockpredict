from unittest import TestCase

from stockpredict.network import StockNeuralNetwork


class TestNetwork(TestCase):

    def test_creation__1(self):
        sn = StockNeuralNetwork(previous=200, future=10)
        self.assertTrue(sn)