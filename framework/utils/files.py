""" 
All frequently used tool functions
"""
import os

def ensure_dir(dir):

    if not os.path.isdir(dir):
        os.makedirs(dir)
        print(f"=> Created dir: {dir}")