import argparse
import copy
import json
import os
import sys

from sample_factory.algorithms.utils.evaluation_config import add_eval_args, add_test_args
from sample_factory.envs.env_config import add_env_args, env_override_defaults
from sample_factory.utils.utils import log, AttrDict, cfg_file, get_git_commit_hash

from sample_factory.utils.utils import str2bool



def get_algo_class(algo):
    algo_class = None

    if algo == 'APPO':
        from sample_factory.algorithms.appo.appo import APPO
        algo_class = APPO
    elif algo == 'DUMMY_SAMPLER':
        from sample_factory.algorithms.dummy_sampler.sampler import DummySampler
        algo_class = DummySampler
    else:
        log.warning('Algorithm %s is not supported', algo)

    return algo_class


def read_cfg_from_json(parser, argv):
    import json
    from sample_factory.utils.utils import AttrDict

    parser.add_argument('--cfg', type=str, default=None, required=False,
                        help='Path to your args without ".json", you will use args in there if you speficfy it.')
    _args, _ = parser.parse_known_args()

    if _args.cfg is None: # Skipping reading if cfg is not specified
        return None

    cfg_path = os.path.join('sample_factory/cfgs/', f'{_args.cfg}.json')
    cfg_from_json = json.load(open(cfg_path, 'r'))
    cfg_from_json = AttrDict(cfg_from_json)

    keys_in_json = list(cfg_from_json.keys())
    cmd_extension = list()
    for key in keys_in_json:
        val = cfg_from_json[key]
        cmd_extension.append( f'--{key}={val}' )

    keys_in_cmd = list()
    for arg in argv:
        key, val = arg.split('=') # Note! val is always a string
        key = key.lstrip('--')
        keys_in_cmd.append(key)

    for key in keys_in_cmd:
        idx_key_in_cmd = keys_in_cmd.index(key)
        if key in keys_in_json:
            idx_key_in_json = keys_in_json.index(key)
            cmd_extension[idx_key_in_json] = argv[idx_key_in_cmd]
        else:
            cmd_extension.append(argv[idx_key_in_cmd])
    return cmd_extension


def arg_parser(argv=None, evaluation=False):
    if argv is None:
        argv = sys.argv[1:]

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    cmd_extension = read_cfg_from_json(parser, argv) # resetting argv by values read from specified cfg file.
    if cmd_extension is not None:
        argv = cmd_extension

    # common args
    parser.add_argument('--algo', type=str, default=None, required=True, help='Algo type to use (pass "APPO" if in doubt)')
    parser.add_argument('--env', type=str, default=None, required=True, help='Fully-qualified environment name in the form envfamily_envname, e.g. atari_breakout or doom_battle')
    parser.add_argument(
        '--experiment', type=str, default=None, required=True,
        help='Unique experiment name. This will also be the name for the experiment folder in the train dir.'
             'If the experiment folder with this name aleady exists the experiment will be RESUMED!'
             'Any parameters passed from command line that do not match the parameters stored in the experiment cfg.json file will be overridden.',
    )
    parser.add_argument(
        '--experiments_root', type=str, default=None, required=False,
        help='If not None, store experiment data in the specified subfolder of train_dir. Useful for groups of experiments (e.g. gridsearch)',
    )
    parser.add_argument('--is_test', type=str2bool, default=False, required=False, help='If True, run experiment in test mode')
    parser.add_argument('-h', '--help', action='store_true', help='Print the help message', required=False)

    basic_args, _ = parser.parse_known_args(argv)
    algo = basic_args.algo
    env = basic_args.env

    # algorithm-specific parameters (e.g. for APPO)
    algo_class = get_algo_class(algo)
    algo_class.add_cli_args(parser)

    # env-specific parameters (e.g. for Doom env)
    add_env_args(env, parser)
    if basic_args.is_test:
        add_test_args(parser)
    if evaluation:
        add_eval_args(parser)

    # env-specific default values for algo parameters (e.g. model size and convolutional head configuration)
    env_override_defaults(env, parser)

    return parser, argv


def parse_args(argv=None, evaluation=False, parser=None):
    if argv is None:
        argv = sys.argv[1:]

    if parser is None:
        parser, argv = arg_parser(argv, evaluation)

    # parse all the arguments (algo, env, and optionally evaluation)
    args = parser.parse_args(argv)

    if args.help:
        parser.print_help()
        sys.exit(0)

    args.command_line = ' '.join(argv)

    # following is the trick to get only the args passed from the command line
    # We copy the parser and set the default value of None for every argument. Since one cannot pass None
    # from command line, we can differentiate between args passed from command line and args that got initialized
    # from their default values. This will allow us later to load missing values from the config file without
    # overriding anything passed from the command line
    no_defaults_parser = copy.deepcopy(parser)
    for arg_name in vars(args).keys():
        no_defaults_parser.set_defaults(**{arg_name: None})
    cli_args = no_defaults_parser.parse_args(argv)

    for arg_name in list(vars(cli_args).keys()):
        if cli_args.__dict__[arg_name] is None:
            del cli_args.__dict__[arg_name]

    args.cli_args = vars(cli_args)
    args.git_hash, args.git_repo_name = get_git_commit_hash()
    return args


def default_cfg(algo='APPO', env='env', experiment='test'):
    """Useful for tests."""
    return parse_args(argv=[f'--algo={algo}', f'--env={env}', f'--experiment={experiment}'])


def load_from_checkpoint(cfg):
    filename = cfg_file(cfg)
    if not os.path.isfile(filename):
        raise Exception(f'Could not load saved parameters for experiment {cfg.experiment}')

    with open(filename, 'r') as json_file:
        json_params = json.load(json_file)
        log.warning('Loading existing experiment configuration from %s', filename)
        loaded_cfg = AttrDict(json_params)

    # override the parameters in config file with values passed from command line
    for key, value in cfg.cli_args.items():
        if key in loaded_cfg and loaded_cfg[key] != value:
            log.debug('Overriding arg %r with value %r passed from command line', key, value)
            loaded_cfg[key] = value

    # incorporate extra CLI parameters that were not present in JSON file
    for key, value in vars(cfg).items():
        if key not in loaded_cfg:
            log.debug('Adding new argument %r=%r that is not in the saved config file!', key, value)
            loaded_cfg[key] = value

    return loaded_cfg


def maybe_load_from_checkpoint(cfg):
    filename = cfg_file(cfg)
    if not os.path.isfile(filename):
        log.warning('Saved parameter configuration for experiment %s not found!', cfg.experiment)
        log.warning('Starting experiment from scratch!')
        return AttrDict(vars(cfg))
    elif not cfg.is_test and not cfg.resume:
        return AttrDict(vars(cfg))
    return load_from_checkpoint(cfg)
