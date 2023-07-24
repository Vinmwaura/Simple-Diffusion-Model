import unittest

import torch
from models.U_Net import U_Net


class TestUNetModel(unittest.TestCase):
    def setUp(self):
        self.u_net = U_Net()
        
        self.num_data = 1
        self.img_C = 3
        self.img_H = 128
        self.img_W = 128

        self.time_val = 1_000

        self.x_T = torch.randn(self.num_data, self.img_C, self.img_H, self.img_W)
        self.time_step = torch.tensor([self.time_val], dtype=torch.long)
    
    def test_unet_forward_pass(self):
        x_0 = self.u_net(self.x_T, self.time_step)
        self.assertEqual(x_0.shape, (self.num_data, self.img_C, self.img_H, self.img_W))
