# -*- coding: utf-8 -*-
"""讓 pytest 不用安裝套件就能 import 專案根目錄的模組。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
