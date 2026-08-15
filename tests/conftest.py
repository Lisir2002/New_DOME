"""pytest 配置：保证从任意位置运行都能导入 sng 包。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
