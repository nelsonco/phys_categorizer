# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:28:19 2020

@author: Court
"""

import os 
import pandas as pd




def NewDataFrame(directory, folder_or_file):
    if folder_or_file == 'folder':
        data = list()
        for folder in os.listdir(directory):
            print(folder)
            for file in os.listdir(directory + '//' + folder):
                if file.endswith(".txt"):
                    category = folder
                    file2 = open(directory+'\\'+folder+'\\'+file, 'r', encoding = 'utf-8')
                    text = file2.read()
                    data_array = [file[:-4], category, text]
                    data.append(data_array)
    else:
        data = list()
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                category = directory.split('\\')[-1]
                file2 = open(directory+'\\'+file, 'r', encoding = 'utf-8')
                text = file2.read()
                data_array = [file[:-4], category, text]
                data.append(data_array)
        
    df = pd.DataFrame(data, columns = ['ID','Category','Text'])
    return df


