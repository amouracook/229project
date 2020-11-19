#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:12:29 2020

@author: amouracook
"""
import numpy as np
import pandas as pd

#%%

# df = pd.read_csv('ahs2017n.csv')
df = pd.read_csv('ahs2017m.csv')

#%%
df_SJ = df.loc[df['OMB13CBSA']=="'41940'"]
df_SJ.to_csv('SJ_41940_Flat.csv')
# df_SF = df.loc[(df['OMB13CBSA']=="'41860'") | (df['OMB13CBSA']=="'41940'") | (df['OMB13CBSA']=="'31080'")]
# df_SF.to_csv('CA_41860_41940_31080_Flat.csv')