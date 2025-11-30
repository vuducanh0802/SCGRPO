# SC GRPO
## Installation

Kinda mess setting up the environment, pray while you do it.

something that I did:

- Since HuggingFace ≥4.41 changed Trainer API. It now always passes the dataset into _get_train_sampler(some_dataset).
In SCGRPO/src/open_r1/trl/trainer/grpo_trainer.py
```
def _get_train_sampler(self) -> Sampler: -> def _get_train_sampler(self, dataset=None) -> Sampler: 
```
- modified /recipes/dra_grpo.yaml: attn_implementation, num_generations, per_device_eval_batch_size, per_device_train_batch_size


Clone the code. We are using the following modulses.

```
module load anaconda3/2023.09-0 
module load git-lfs/3.3.0 
module load cuda/11.8.0 

```

Please follow the instructions of [Open-RS](https://github.com/knoveleng/open-rs) to install the environment.
Log in to Hugging Face and Weights & Biases:
```
huggingface-cli login
wandb login
```

```
source activate openr3
```

**You can then remove ```trl``` package from the environment, because we customized it.**



## Training

### DRA-GRPO
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 11188 \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  src/open_r1/grpo.py \
  --config recipes/dra_grpo.yaml 
```


### DRA-DR. GRPO
```
ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 18007 \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  src/open_r1/drgrpo.py \
  --config recipes/dra_dr_grpo.yaml
```

All weights will update to Huggingface.

## Inference via lighteval (Test multiple steps)
We have an evaluation template 

```
base evaL_all.sh
```

## Checkpoints (Updated in [Huggingface Repo](https://huggingface.co/SpiceRL)  )
```
MODEL= xxx
MODEL_NAME=$(basename "$MODEL")

TASKS="math_500 amc23 minerva olympiadbench aime24"

OUTPUT_DIR=data-test/evals/${MODEL_NAME}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
for TASK in $TASKS; do
  lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR"
done

done
```

Replace ```xxx``` with  ```SpiceRL/DRA-GRPO``` or ```SpiceRL/DRA-DR.GRPO```. The evaluation only requires one GPU.

##

Our code is built based on [Open-rs](https://github.com/knoveleng/open-rs). Thanks!



