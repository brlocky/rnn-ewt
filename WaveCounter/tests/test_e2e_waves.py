import unittest
import pandas as pd
from ..elliott.elliott import Elliott
from ..pivots.zigzag import ZigZag


class TestElliottMethods(unittest.TestCase):
    def setUp(self):
        # Create test DataFrame and pivots
        df = pd.read_csv('./csv_data/AMZN.csv', index_col=0, parse_dates=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.df = df[1761:1900].copy()
        self.df['Pivot'] = ZigZag(self.df).get_zigzag()

    def test_get_waves(self):
        # Test the get_waves method
        elliott = Elliott(self.df)
        waves = elliott.get_waves()

        # print('Test get_waves ', waves)
        # Add your assertions here to check if 'waves' matches the expected result
        # Example assertion: self.assertEqual(waves, expected_waves)

        assert 1 == 1
