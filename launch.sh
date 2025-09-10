#!/bin/bash

export "RUN_NAME=$1"
accelerate launch --config_file ./accelerate_config.yaml ./gemma3_finetune.py