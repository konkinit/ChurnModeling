from sys import path
from os import getcwd

if getcwd() not in path:
    path.append(getcwd())