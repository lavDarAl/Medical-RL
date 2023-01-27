import argparse
from argparse import Namespace

def parseArguments() -> Namespace:
    parser = argparse.ArgumentParser("Perform Medical Reinforcement Learning")
    parser.add_argument('--freeze_encoder', action='store_true', help="Freeze Encoder Language Model")
    parser.add_argument('--use_action_embeddings', action='store_true', help="Add Embedding Layer for actions that is multiplied with observation encoding")
    parser.add_argument("--max_episodes", type=int, default=100, help="Number of Episodes to run.")
    parser.add_argument("--max_diseases", type=int, default=None, help="Number of Diseases to keep in environment.")
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker agents to use for training")
    parser.add_argument("--sequence_length", type=int, default=128, help="length of observation sequences")
    parser.add_argument("--rollout_fragment_length", type=int, default=16, help="number of rollouts/size of replay buffer")
    parser.add_argument("--inference_batch_size", type=int, default=8, help="Number of environments per worker")
    parser.add_argument("--batch_size", type=int, default=8, help="Train batch size sent to GPU")
    parser.add_argument("--output_hidden_layers", type=int, default=0, help="Number of hidden layers for network output heads")
    parser.add_argument("--output_hidden_size", type=int, default=1024, help="Hidden Size of output heads if using multilayer output")
    parser.add_argument("--num_gpu_per_worker", type=float, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--replay_proportion", type=float, default=0.1, help="Set >0 to enable experience replay with p:1 ratio")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="Transformer Encoder to use.")
    parser.add_argument("--data_path", type=str, default="../Medical-Gym/data/project_hospital", help="Path to Environment data")
    parser.add_argument("--action_embedding_path", type=str, default="../Medical-Gym/data/project_hospital", help="Path to Environment data")
    parser.add_argument("--log_level", type=str, default="WARN", help="Log Level to use")
    parser.add_argument("--log_dir", type=str, default="~/ray-results", help="Directory for Ray logs")
    return parser.parse_known_args()[0]