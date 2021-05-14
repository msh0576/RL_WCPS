# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:36:09 2020

@author: Sihoon
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--verbosity', help = 'increase output verbosity')
args = parser.parse_args()

if args.verbosity:
    print("verbosity turned on")