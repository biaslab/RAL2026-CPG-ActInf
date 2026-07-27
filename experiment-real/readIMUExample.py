#!/usr/bin/python3
#  -*- coding: UTF-8 -*-

# MindPlus
# Python
from PetoiRobot import *

logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

goodPorts = {}

try:

    connectPort(goodPorts)    

    logger.info(f'Reading IMU ..')
    serial_IMU = send(goodPorts, ['v', 0])
    ypr_acc = serial_IMU[1].split()
    logger.info(ypr_acc)    

    logger.info(f'Sleeping .. ')
    time.sleep(2.0)

    closeAllSerial(goodPorts)

except:
    closeAllSerial(goodPorts)
