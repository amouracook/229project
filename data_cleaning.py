#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 08:34:28 2020

@author: tamrazovd
"""


import pandas as pd
import numpy as np

#%%

df = pd.read_csv('ahs2017m.csv')

#%%
df_SF = df.loc[(df['OMB13CBSA']=="'41860'") | (df['OMB13CBSA']=="'41940'") | (df['OMB13CBSA']=="'31080'")]
df_SF.to_csv('CA_41860_41940_31080_Flat.csv')
