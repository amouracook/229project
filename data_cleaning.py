#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 08:34:28 2020

@author: tamrazovd
"""


import pandas as pd
import numpy as np

#%%

df = pd.read_csv('ahs2017n.csv')

#%%
df_SF = df.loc[df['OMB13CBSA']=="'41860'"]
df_SF.to_csv('SF_41860_Flat.csv')
