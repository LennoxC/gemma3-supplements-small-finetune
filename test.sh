#!/bin/bash

export "TEST_RUN_NAME=$1"
#accelerate launch --config_file ./accelerate_config_test.yaml ./gemma3_test.py
python ./gemma3_test.py