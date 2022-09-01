import unittest

import torch
from models.U_Net import U_Net


class TestUNetModel(unittest.TestCase):
    def setUp(self):
        self.u_net = U_Net()
        self.x_T = torch.randn(16, 3, 128, 128)
        self.time_step = torch.tensor([1000], dtype=torch.long)
    
    def test_unet_forward_pass(self):
        x_0 = self.u_net(self.x_T, self.time_step)
        self.assertEqual(x_0.shape, (16, 3, 128, 128))
