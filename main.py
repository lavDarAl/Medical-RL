import ray
import ray.rllib.algorithms.impala as IMPALA
import gym
import torch
import json
import os

from argparse import Namespace
from typing import Dict
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models import ModelCatalog
from tqdm import tqdm
from tensorboardX import SummaryWriter
from arguments import parseArguments
from model import IMPALATransformer


def build_config(args: Namespace) -> Dict:
    config                                     = IMPALA.DEFAULT_CONFIG.copy()
    config["framework"]                        = "torch"
    config["num_gpus"]                         = args.num_gpu
    config["num_multi_gpu_tower_stacks"]       = 3
    config["num_workers"]                      = args.num_workers
    config["num_envs_per_worker"]              = args.inference_batch_size
    config["num_cpus_per_worker"]              = args.inference_batch_size + 1 #if args.num_gpu_per_worker > 0 else args.num_workers * 24
    config["num_gpus_per_worker"]              = args.num_gpu_per_worker
    config["rollout_fragment_length"]          = args.rollout_fragment_length
    config["train_batch_size"]                 = args.batch_size
    config["replay_proportion"]                = args.replay_proportion
    config["replay_buffer_num_slots"]          = 128
    config["env"]                              = "gym_medical:doctorsim-v0"
    config["log_level"]                        = args.log_level
    config["env_config"]["observation_length"] = args.sequence_length
    config["env_config"]["max_diseases"]       = args.max_diseases
    config["env_config"]["tokenizer"]          = args.model_name_or_path
    config["env_config"]["data_path"]          = args.data_path
    config["remote_worker_envs"]               = False
    config["exploration_config"]               = {"type": "EpsilonGreedy"}
    #config["remote_env_batch_wait_ms"]         = 20 

    model_config                        = MODEL_DEFAULTS.copy()
    model_config["custom_model"]        = "IMPALATransformer"
    model_config["custom_model_config"] = {
        "model_name_or_path": args.model_name_or_path,
        "freeze_encoder": args.freeze_encoder,
        "output_hidden_layers": args.output_hidden_layers,
        "output_hidden_size": args.output_hidden_size,
        "action_embeddings": None
    }
    config["model"] = model_config

    return config


if __name__ == "__main__":

    args   = parseArguments()

    if args.action_embedding_path and args.action_embedding_path != "None":
        tempgym = gym.make(
            "gym_medical:doctorsim-v0", 
            data_path=args.data_path, 
            max_diseases = args.max_diseases,  
            tokenizer = args.model_name_or_path,
        )
        action_embeddings = torch.load("args.action_embedding_path")
        args.action_embeddings = action_embeddings[tempgym.procedures_keys, :]


    config = build_config(args)


    ray.init(num_cpus=256)

    ModelCatalog.register_custom_model("IMPALATransformer", IMPALATransformer)

    # trainer = IMPALA.ImpalaTrainer(
    #     config = config,
    #     env    = "gym_medical:doctorsim-v0",
    # )

    stop = {
        "episodes_total": args.max_episodes
    }

    print(IMPALA.ImpalaTrainer.default_resource_request(config=config)._bundles)

    analysis = ray.tune.run(
        IMPALA.ImpalaTrainer,
        config=config,
        stop = stop,
        local_dir=args.log_dir,
        reuse_actors=True,
        checkpoint_at_end=True
    )

    for t in analysis.trials:
        #w = SummaryWriter(os.path.dirname(t.logdir))
        #w.add_hparams(t.config, {})
        obj = {str(k):str(v) for k,v in t.config.items()}
        path = open(os.path.join(os.path.dirname(t.logdir), "config.json"), "w")
        json.dump(obj, path)

    ray.shutdown()
