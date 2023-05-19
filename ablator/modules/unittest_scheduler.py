import torch
import unittest
import sys
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append('/app/')

from ablator.modules.scheduler import (
    SchedulerConfig,
    OneCycleConfig,
    PlateuaConfig,
    StepLRConfig,
)

class TestSchedulers(unittest.TestCase):

    def __str__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"

    
    def test_onecycle_scheduler(self):
        config = SchedulerConfig(name='cycle', arguments={
            'max_lr': 0.1,
            'total_steps': 100,
            'step_when': 'train'
        })
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = config.make_scheduler(model, optimizer)
        
        inputs = (model, optimizer)
        outputs = scheduler
        expected_output_type = torch.optim.lr_scheduler.OneCycleLR
        
        print(f"Inputs: {inputs}")
        print(f"Outputs: {str(outputs)}")
        print(f"Expected Output Type: {str(expected_output_type)}")
        
        self.assertIsInstance(scheduler, expected_output_type)
    
    def test_plateau_scheduler(self):
        config = SchedulerConfig(name='plateau', arguments={
            'step_when': 'val',
            'patience': 3,
            'min_lr': 1e-6,
            'mode': 'min',
            'factor': 0.5,
            'threshold': 1e-3,
            'verbose': False,
        })
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = config.make_scheduler(model, optimizer)
        
        inputs = (model, optimizer)
        outputs = scheduler
        expected_output_type = torch.optim.lr_scheduler.ReduceLROnPlateau
        
        print(f"Inputs: {inputs}")
        print(f"Outputs: {str(outputs)}")
        print(f"Expected Output Type: {str(expected_output_type)}")
        
        self.assertIsInstance(scheduler, expected_output_type)
    
    def test_steplr_scheduler(self):
        config = SchedulerConfig(name='step', arguments={
            'step_when': 'epoch',
            'step_size': 2,
            'gamma': 0.5
        })
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = config.make_scheduler(model, optimizer)
        
        inputs = (model, optimizer)
        outputs = scheduler
        expected_output_type = torch.optim.lr_scheduler.StepLR
        
        print(f"Inputs: {inputs}")
        print(f"Outputs: {str(outputs)}")
        print(f"Expected Output Type: {str(expected_output_type)}")
        
        self.assertIsInstance(scheduler, expected_output_type)

if __name__ == "__main__":
    unittest.main()