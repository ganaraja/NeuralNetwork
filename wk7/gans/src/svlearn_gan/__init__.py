#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
from svlearn.config.configuration import ConfigurationMixin
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

config = ConfigurationMixin().load_config()
current_task = config['current_task']

#  -------------------------------------------------------------------------------------------------
#  Enums
#  -------------------------------------------------------------------------------------------------
class Task(Enum):
    MNIST = 'mnist-classification'
    TREE = 'tree-classification'


__all__ = ['config', 'current_task', 'Task']
