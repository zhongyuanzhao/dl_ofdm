#! /usr/bin/python
######################################################################################
# Libraries for local helper functions
# Author: Zhongyuan Zhao 
# Date: 2021-03-10
# Link: https://github.com/zhongyuanzhao/dl_ofdm
# Cite this work:
# Zhongyuan Zhao, Mehmet C. Vuran, Fujuan Guo, and Stephen Scott, "Deep-Waveform: 
# A Learned OFDM Receiver Based on Deep Complex-valued Convolutional Networks," 
# EESS.SP, vol abs/1810.07181, Mar. 2021, [Online] https://arxiv.org/abs/1810.07181
# 
# Copyright (c) 2021: Zhongyuan Zhao
# Houston, Texas, United States
# <zhongyuan.zhao@huskers.unl.edu>
######################################################################################

import sys
import os
import re
import subprocess
import numpy as np
from numpy import genfromtxt
import time

cwd = os.getcwd()


def runlocalpython(parameters, pyfile):
    commands = pyfile + ' ' + parameters
    filenametag = commands  
    filenametag = re.sub('[^0-9a-zA-Z]+', '_', filenametag)
    filenametag = filenametag[-180:]

    print("Start: %s"%(filenametag))
    subprocess.call("python -u "+commands + " > %s.out"%(filenametag), shell=True)
    time.sleep(0.5)
    print('Job finished: %s' % (commands))
    return True


