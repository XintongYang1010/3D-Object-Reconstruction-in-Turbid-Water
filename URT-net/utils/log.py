#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 13:46
# @Author  : jc Han
# @help    :
import datetime

def save_log(file_path,info):
    with open(file_path, 'a') as file:

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        file.write(log_entry)
