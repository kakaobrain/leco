import os
import shutil
import sys

from sample_factory.algorithms.utils.arguments import maybe_load_from_checkpoint, get_algo_class, parse_args


def run_algorithm(cfg):
    cfg = maybe_load_from_checkpoint(cfg)

    if 'record_to' in cfg and cfg.record_to is not None:
        if os.path.isdir(os.path.join(cfg.train_dir, cfg.experiment, cfg.record_to)):
            print('recording directory already exists! remove directory... ')
            shutil.rmtree(os.path.join(cfg.train_dir, cfg.experiment, cfg.record_to))
            print('done!')

    algo = get_algo_class(cfg.algo)(cfg)
    algo.initialize()
    status = algo.run()
    algo.finalize()
    return status


def main():
    """Script entry point."""
    cfg = parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
