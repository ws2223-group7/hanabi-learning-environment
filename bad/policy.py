# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, unnecessary-pass
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.actionnetwork import ActionNetwork

class Policy:
    '''policy'''
    def __init__(self, network: ActionNetwork) -> None:
        '''init'''
        pass
