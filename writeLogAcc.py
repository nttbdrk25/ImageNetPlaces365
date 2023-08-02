# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:35:11 2022

@author: tuann
"""
import os.path
from datetime import datetime

def writeLogAcc(filename='LogAcc1.txt', strtext=''):
    if not os.path.exists(filename):
        file1 = open(filename,"w")
    else:
        file1 = open(filename,"a")
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    strtext = dt_string + ' ' + strtext + '\n'
    file1.writelines(strtext)
    file1.close() #to change file access modes