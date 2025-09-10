#!/bin/bash

source .env
tensorboard --logdir=$LOGS_HOME --port=$1
