"""Main module."""
import argparse
import logging

def snn(hyperparams: dict):
    return None


def main(args):
    parser = argparse.ArgumentParser(description="tabular_neural_net")
    parser.add_argument("--n_epoch", required=True, help="...")
    parser.add_argument("--batch_size", required=True, help="...")
    parser.add_argument("--layer_dropout", required=True, help="...")
    parser.add_argument("--embed_dropout", required=True, help="...")
    parser.add_argument("--layer_1_neurons", required=True, help="...")
    parser.add_argument("--layer_2_neurons", required=True, help="...")
    parser.add_argument("--layer_3_neurons", required=True, help="...")
    parser.add_argument("--artifact_dir", required=True, help="...")
    parser.add_argument("--shapley_interpretation", required=True, help="...")
    parser.add_argument("--siamese_learner", required=True, help="...")
    args = parser.parse_args(args[1:])

    hyperparams = {
        "n_epoch": eval(args.n_epoch),
        "batch_size": eval(args.batch_size),
        "layer_dropout": eval(args.layer_dropout),
        "embed_dropout": eval(args.embed_dropout),
        "layer_1_neurons": eval(args.layer_1_neurons),
        "layer_2_neurons": eval(args.layer_2_neurons),
        "layer_3_neurons": eval(args.layer_3_neurons),
        "target_investment_amount": eval(args.target_investment_amount),
        "artifact_dir": args.artifact_dir,
        "shapley_interpretation": eval(args.shapley_interpretation),
        "siamese_learner": eval(args.siamese_learner),
    }

    snn(hyperparams)
    logging.info("Job complete!")


if __name__ == "__main__":
    import sys
    import torch
    torch.set_num_threads(1)
    sys.exit(main(sys.argv))
