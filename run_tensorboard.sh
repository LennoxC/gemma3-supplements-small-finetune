#!/bin/bash

source .env
tensorboard --logdir=$LOGS_DIR --port=$1
