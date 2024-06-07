ACCELERATE_LOG_LEVEL=info 
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
 scripts/run_cpo.py \
 training_configs/llama-3-8b-instruct-cpo.yaml