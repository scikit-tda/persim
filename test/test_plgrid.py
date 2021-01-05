#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:26:54 2020

@author: gabrielleangeloro
"""
import unittest 
import numpy as np

from PersistenceLandscapeGrid import PersistenceLandscapeGrid


class TestPersistenceLandscapeGrid(unittest.TestCase):
    
    
    def test_pl_funct_values(self):
        """
        Test PersistenceLandscape
        """
        diagrams=[np.array([[2,6],[4,10]])]
        P1 = PersistenceLandscapeGrid(0, 10, 11, diagrams,
                                     homological_degree=0)
        P2 = PersistenceLandscapeGrid(0, 10, 6, diagrams,
                                     homological_degree=0)
        P3 = PersistenceLandscapeGrid(0, 10, 21, diagrams,
                                     homological_degree=0)
        
        P1.compute_landscape()
        P2.compute_landscape()
        P3.compute_landscape()
        
        
        self.assertEqual(P1.funct_values, 
                np.array([[0., 0., 0., 1., 2., 1., 2., 3., 2., 1., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]] )).all()
        
        self.assertEqual(P2.funct_values, np.array([[0., 0., 2.0, 2.0, 2.0, 0.]]))
        
        self.assertEqual(P3.funct_values, 
        np.array([[0. , 0. , 0. , 0. , 0. , 0.5, 1. , 1.5, 2. , 1.5, 1. , 1.5, 2. ,
        2.5, 3. , 2.5, 2. , 1.5, 1. , 0.5, 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 1. , 0.5, 0. ,
        0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]]))
        

if __name__ == '__main__':
    unittest.main()