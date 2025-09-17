
# the first name is the name appear in beaker
# for more details, do `python -m olmo_core.launch.beaker --help`

# basically it's running `src/examples/llm/train.py`
# the first config is a run name (used for save_folder, wandb name, etc)
# for more details, `python src/examples/llm/train.py olmo1B-pretrain-01 --dry-run`

# -- trainer.load_path if you want to load from another model

# when the config is a class, we could either use a json string or set individual value
# e.g., `--trainer.hard_stop='value: 100, unit: steps'` or 
#       `--trainer.hard_stop.value=100 --trainer.hard_stop.unit=steps`

python -m olmo_core.launch.beaker \
	--name olmo1B-pretrain \
	--gpus 2 \
	--nodes 1 \
	--budget ai2/oe-base \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
	--priority high \
	--preemptible \
	--allow-dirty \
	--weka=oe-training-default \
	  -- src/examples/llm/train.py \
		olmo1B-pretrain-01 \
		--trainer.save_folder=/oe-training-default/sewonm \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.project=sewonm \
		--trainer.callbacks.lm_evaluator.enabled=false \
	    	--trainer.callbacks.downstream_evaluator.enabled=false \
		--trainer.no_checkpoints \
		--trainer.hard_stop='{value: 100, unit: steps}'


