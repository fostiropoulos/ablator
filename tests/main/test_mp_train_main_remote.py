import unittest
from unittest.mock import Mock, patch
from ablator.main.mp import train_main_remote, TrialState
from ablator.main.model.main import CheckpointNotFoundError
from ablator.modules.metrics.main import LossDivergedError

class TestTrainMainRemote(unittest.TestCase):

    def setUp(self):
        # Mocking required parameters for train_main_remote function
        self.model = Mock()
        self.run_config = Mock()
        self.mp_logger = Mock()
        self.root_dir = Mock()

    @patch('ablator.main.mp.train_main_remote')
    def test_successful_training(self, mock_train_main_remote):
        # Mocking successful training
        self.model.train.return_value = {}
        
        config, metrics, state = train_main_remote(self.model, self.run_config, self.mp_logger, self.root_dir)
        
        self.assertEqual(state, TrialState.COMPLETE)

    @patch('ablator.main.mp.train_main_remote')
    def test_training_with_loss_diverged_error(self, mock_train_main_remote):
        # Mocking training to raise LossDivergedError
        self.model.train.side_effect = LossDivergedError
        
        config, metrics, state = train_main_remote(self.model, self.run_config, self.mp_logger, self.root_dir)
        
        self.assertEqual(state, TrialState.PRUNED_POOR_PERFORMANCE)

    @patch('ablator.main.mp.train_main_remote')
    def test_training_with_checkpoint_not_found_error(self, mock_train_main_remote):
        # Mocking training to raise CheckpointNotFoundError
        self.model.train.side_effect = CheckpointNotFoundError
        
        config, metrics, state = train_main_remote(self.model, self.run_config, self.mp_logger, self.root_dir, clean_reset=True)
        
        self.assertEqual(state, TrialState.RECOVERABLE_ERROR)

if __name__ == '__main__':
    unittest.main()
