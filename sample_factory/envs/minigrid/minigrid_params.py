from sample_factory.utils.utils import str2bool

# noinspection PyUnusedLocal
def minigrid_override_defaults(env, parser):
    parser.set_defaults(
        encoder_subtype='minigrid_convnet_tiny',
        hidden_size=128,
        obs_subtract_mean=4.0,
        obs_scale=8.0,
        exploration_loss_coeff=0.005,
        env_frameskip=1,
    )

def add_minigrid_env_args(env, parser):
    p = parser

    p.add_argument(
        '--minigrid_one_task_per_worker', default=False, type=str2bool,
        help='By default SampleFactory will run several tasks per worker. E.g. if num_envs_per_worker=30 then each and every worker'
             'will run all 30 tasks of DMLab-30. In such regime an equal amount of samples will be collected for all tasks'
             'throughout training. This can potentially limit the throughput, because in this case the system is forced to'
             'collect the same number of samples from slow and from fast environments (and the simulation speeds vary greatly, especially on CPU)'
             'This flag enables a different regime, where each worker is focused on a single task. In this case the total number of workers should'
             'be a multiple of 30 (for DMLab-30), and e.g. 17th task will be executed on 17th, 47th, 77th... worker',
    )
    p.add_argument('--random_action', default=False, type=str2bool, help='Whether to use random action for rendering')