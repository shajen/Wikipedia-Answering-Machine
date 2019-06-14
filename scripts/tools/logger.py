#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

def configLogger(verbose):
	levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
	level = levels[min(len(levels)-1, verbose)]
	logging.basicConfig(format='[%(asctime)s][%(levelname)7s] %(message)s', level=level, datefmt="%Y-%m-%d %H:%M:%S")
