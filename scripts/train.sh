
# first do `beaker config set default_workspace ai2/flex2`

# do `python src/scripts/train/OLMoE-1B-7B.py dry_run [runname] ai2/jupiter`
# for overriding hyperparams

# by default, it uses ai2/flex2 workspace, and uses /weka/oe-training-default/ai2-llm/{your username}/{name}

# training dataset default is
# `dataset.mix_base_dir=/weka/oe-training-default/ai2-llm --dataset.mix=OLMoE-mix-0824` 
# TODO on how to change

name=moe_default

python src/scripts/train/OLMoE-1B-7B.py launch \
	$name \
	ai2/jupiter \
	--launch.num_nodes=4 \
	--launch.task_name=$name \
	--launch.num_nodes=1 \
	--launch.priority=urgent \
	--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
	--trainer.callbacks.wandb='{enabled: true, entity: sewonm, project: olmoe}' \
	--trainer.callbacks.lm_evaluator.enabled=false \
	--trainer.callbacks.downstream_evaluator.enabled=false \
	--trainer.no_checkpoints \
	--trainer.hard_stop='{value: 100, unit: steps}'



