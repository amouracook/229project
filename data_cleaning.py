#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:08:45 2020

@author: Aaron
"""

import numpy as np
import pandas as pd

#%%

df = pd.read_csv('ahs2017n.csv')
df = pd.read_csv('ahs2017m.csv')

#%%
df_SF = df.loc[df['OMB13CBSA']=="'41860'"]
df_SF.to_csv('SF_41860_Flat.csv')
df_SF = df.loc[(df['OMB13CBSA']=="'41860'") | (df['OMB13CBSA']=="'41940'") | (df['OMB13CBSA']=="'31080'")]
df_SF.to_csv('CA_41860_41940_31080_Flat.csv')