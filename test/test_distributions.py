import unittest
from helpers.preprocessing import cross_entropy_weights, get_distribution, read_all_data
from sklearn.model_selection import train_test_split
import math
import os

if os.getcwd().split('/')[-1] == 'test':
    os.chdir('..')


class TestDistribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dict = read_all_data()
        # cls.ann = cls.data_dict['ann']['labels'].tolist()
        cls.dist_dict = get_distribution(cls.data_dict['ann']['labels'].tolist())

    @staticmethod
    def tolerance(lst, tol):
        '''
        Check if all values in lst are equal within tolerance tol
        '''
        for i in range(len(lst)-1):
            if not math.isclose(lst[i], lst[i+1], rel_tol=tol, abs_tol=tol):
                return False
        return True
    
    def test_cross_entropy_weights(self):
        weights = cross_entropy_weights(self.dist_dict['fracs'])
        overall_counts = self.dist_dict['counts']
        out = [w*c for w, c in zip(weights.tolist(), overall_counts)]
        self.assertTrue(TestDistribution.tolerance(out, 0.00001))
        
    def test_train_val_split_dist(self):
        # Verify split shares same dist
        imu = self.data_dict['imu'].to_numpy()
        ann = self.data_dict['ann'].to_numpy().flatten()
        _, _, y_train, y_val = train_test_split(imu, ann, test_size=0.2, random_state=42)
        
        train_dist = get_distribution(y_train.tolist())['fracs']
        val_dist = get_distribution(y_val.tolist())['fracs']
        self.assertTrue(
            all(
                [TestDistribution.tolerance([t, v, d], tol=0.001) 
                 for t, v, d in zip(train_dist, val_dist, self.dist_dict['fracs'])]
            )
        )
