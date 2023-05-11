import sys
import time
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import os



from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict




def checkpoint_convert_from_torchbeast_to_sf(checkpoint_dict, torchbeast_checkpoint):

    # for k, v in checkpoint_dict['model'].items():
    #     print(k)
    #     print(v.size())

    checkpoint_dict['model']['encoder.conv_head.0.weight'] = torchbeast_checkpoint['model_state_dict']['feat_convs.0.0.weight']
    checkpoint_dict['model']['encoder.conv_head.0.bias'] = torchbeast_checkpoint['model_state_dict']['feat_convs.0.0.bias']
    checkpoint_dict['model']['encoder.conv_head.4.weight'] = torchbeast_checkpoint['model_state_dict']['feat_convs.1.0.weight']
    checkpoint_dict['model']['encoder.conv_head.4.bias'] = torchbeast_checkpoint['model_state_dict']['feat_convs.1.0.bias']
    checkpoint_dict['model']['encoder.conv_head.8.weight'] = torchbeast_checkpoint['model_state_dict']['feat_convs.2.0.weight']
    checkpoint_dict['model']['encoder.conv_head.8.bias'] = torchbeast_checkpoint['model_state_dict']['feat_convs.2.0.bias']

    # 16, 16, 3, 3 / 16,16,3,3
    checkpoint_dict['model']['encoder.conv_head.2.res_block_core.1.weight'] = torchbeast_checkpoint['model_state_dict']['resnet1.0.1.weight']
    checkpoint_dict['model']['encoder.conv_head.2.res_block_core.1.bias'] = torchbeast_checkpoint['model_state_dict']['resnet1.0.1.bias']
    checkpoint_dict['model']['encoder.conv_head.2.res_block_core.3.weight'] = torchbeast_checkpoint['model_state_dict']['resnet1.0.3.weight']
    checkpoint_dict['model']['encoder.conv_head.2.res_block_core.3.bias'] = torchbeast_checkpoint['model_state_dict']['resnet1.0.3.bias']

    # 16, 16, 3, 3 / 32, 32, 3, 3
    checkpoint_dict['model']['encoder.conv_head.3.res_block_core.1.weight'] = torchbeast_checkpoint['model_state_dict']['resnet2.0.1.weight']
    checkpoint_dict['model']['encoder.conv_head.3.res_block_core.1.bias'] = torchbeast_checkpoint['model_state_dict']['resnet2.0.1.bias']
    checkpoint_dict['model']['encoder.conv_head.3.res_block_core.3.weight'] = torchbeast_checkpoint['model_state_dict']['resnet2.0.3.weight']
    checkpoint_dict['model']['encoder.conv_head.3.res_block_core.3.bias'] = torchbeast_checkpoint['model_state_dict']['resnet2.0.3.bias']

    # 32, 32, 3, 3 / 32, 32, 3, 3
    checkpoint_dict['model']['encoder.conv_head.6.res_block_core.1.weight'] = torchbeast_checkpoint['model_state_dict']['resnet1.1.1.weight']
    checkpoint_dict['model']['encoder.conv_head.6.res_block_core.1.bias'] = torchbeast_checkpoint['model_state_dict']['resnet1.1.1.bias']
    checkpoint_dict['model']['encoder.conv_head.6.res_block_core.3.weight'] = torchbeast_checkpoint['model_state_dict']['resnet1.1.3.weight']
    checkpoint_dict['model']['encoder.conv_head.6.res_block_core.3.bias'] = torchbeast_checkpoint['model_state_dict']['resnet1.1.3.bias']

    # 32, 32, 3, 3 / 16, 16, 3, 3
    checkpoint_dict['model']['encoder.conv_head.7.res_block_core.1.weight'] = torchbeast_checkpoint['model_state_dict']['resnet2.1.1.weight']
    checkpoint_dict['model']['encoder.conv_head.7.res_block_core.1.bias'] = torchbeast_checkpoint['model_state_dict']['resnet2.1.1.bias']
    checkpoint_dict['model']['encoder.conv_head.7.res_block_core.3.weight'] = torchbeast_checkpoint['model_state_dict']['resnet2.1.3.weight']
    checkpoint_dict['model']['encoder.conv_head.7.res_block_core.3.bias'] = torchbeast_checkpoint['model_state_dict']['resnet2.1.3.bias']

    # 32, 32, 3, 3 / 32, 32, 3, 3
    checkpoint_dict['model']['encoder.conv_head.10.res_block_core.1.weight'] = torchbeast_checkpoint['model_state_dict']['resnet1.2.1.weight']
    checkpoint_dict['model']['encoder.conv_head.10.res_block_core.1.bias'] = torchbeast_checkpoint['model_state_dict']['resnet1.2.1.bias']
    checkpoint_dict['model']['encoder.conv_head.10.res_block_core.3.weight'] = torchbeast_checkpoint['model_state_dict']['resnet1.2.3.weight']
    checkpoint_dict['model']['encoder.conv_head.10.res_block_core.3.bias'] = torchbeast_checkpoint['model_state_dict']['resnet1.2.3.bias']

    # 32, 32, 3, 3, / 32, 32, 3, 3
    checkpoint_dict['model']['encoder.conv_head.11.res_block_core.1.weight'] = torchbeast_checkpoint['model_state_dict']['resnet2.2.1.weight']
    checkpoint_dict['model']['encoder.conv_head.11.res_block_core.1.bias'] = torchbeast_checkpoint['model_state_dict']['resnet2.2.1.bias']
    checkpoint_dict['model']['encoder.conv_head.11.res_block_core.3.weight'] = torchbeast_checkpoint['model_state_dict']['resnet2.2.3.weight']
    checkpoint_dict['model']['encoder.conv_head.11.res_block_core.3.bias'] = torchbeast_checkpoint['model_state_dict']['resnet2.2.3.bias']

    checkpoint_dict['model']['encoder.fc_after_enc.0.weight'] = torchbeast_checkpoint['model_state_dict']['fc.weight']
    checkpoint_dict['model']['encoder.fc_after_enc.0.bias'] = torchbeast_checkpoint['model_state_dict']['fc.bias']
    checkpoint_dict['model']['critic_linear.weight'] = torchbeast_checkpoint['model_state_dict']['baseline.weight']
    checkpoint_dict['model']['critic_linear.bias'] = torchbeast_checkpoint['model_state_dict']['baseline.bias']
    checkpoint_dict['model']['action_parameterization.distribution_linear.weight'] = torchbeast_checkpoint['model_state_dict']['policy.weight']
    checkpoint_dict['model']['action_parameterization.distribution_linear.bias'] = torchbeast_checkpoint['model_state_dict']['policy.bias']

    return checkpoint_dict


def enjoy(cfg, max_num_frames=1e9, torchbeast_ckpt_path=None,
          image_folder_name=None, random=False):
    cfg = load_from_checkpoint(cfg)

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 4  # for evaluation
    cfg.num_envs = 1

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
    env.seed(0)


    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    #device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    device = torch.device('cpu')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)

    # if torchbeast_ckpt_path is not None:
    #     torchbeast_checkpoint = torch.load(torchbeast_ckpt_path)
    #     checkpoint_dict = checkpoint_convert_from_torchbeast_to_sf(checkpoint_dict, torchbeast_checkpoint)

    if not random:
        actor_critic.load_state_dict(checkpoint_dict['model'])

    # eval mode
    actor_critic.eval()

    episode_rewards = [deque([], maxlen=1) for _ in range(env.num_agents)]
    true_rewards = [deque([], maxlen=1) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    obs = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = np.zeros(env.num_agents)
    finished_episode = [False] * env.num_agents

    raw_rew = torch.zeros(1, 1, device=device)
    # This supports only single-tensor actions ATM.
    actions = torch.zeros(1, 1, dtype=torch.int64)
    done_count = 0

    ablation_list = []

    ii = 0
    task_id = 0
    if not os.path.exists(image_folder_name):
        os.mkdir(image_folder_name)
    with torch.no_grad():
        while not max_frames_reached(num_frames):
            obs_torch = AttrDict(transform_dict_observations(obs))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()
            policy_outputs = actor_critic(obs_torch, actions, torch.Tensor([raw_rew[0]]).to(device), rnn_states, task_id, with_action_distribution=True)


            # sample actions from the distribution by default
            actions = policy_outputs.actions

            action_distribution = policy_outputs.action_distribution
            if isinstance(action_distribution, ContinuousActionDistribution):
                if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                    actions = action_distribution.means

            actions = actions.cpu().numpy()

            rnn_states = policy_outputs.rnn_states

            #for _ in range(render_action_repeat):
            for _ in range(1):
                if not cfg.no_render:
                    target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                    current_delay = time.time() - last_render_start
                    time_wait = target_delay - current_delay

                    if time_wait > 0:
                        # log.info('Wait time %.3f', time_wait)
                        time.sleep(time_wait)

                    last_render_start = time.time()
                    #env.render()
                    image = env.render(mode='rgb_array')
                    image = Image.fromarray(image, 'RGB')
                    image.save(f'./{image_folder_name}/{ii:04d}.png')
                    ii += 1
                    print(ii)



                #actions = [1]
                obs, rew, done, infos, raw_rew = env.step(actions)


                # if raw_rew[0] > 0:
                #     print(raw_rew[0])
                #     print(actions)

                episode_reward += raw_rew
                num_frames += infos[0]['num_frames']
                ablation_list.append({'step': num_frames, 'action': actions, 'rew': rew, 'raw_rew': raw_rew, 'episode_reward': episode_reward[0],
                                      'lives': env.unwrapped.ale.lives()})

                for agent_i, done_flag in enumerate(done):
                    if done_flag:
                        done_count += 1
                        finished_episode[agent_i] = True
                        episode_rewards[agent_i].append(episode_reward[agent_i])
                        true_rewards[agent_i].append(infos[agent_i].get('true_reward', episode_reward[agent_i]))
                        log.info('Episode finished for agent %d at %d frames. Reward: %.3f, true_reward: %.3f', agent_i, num_frames, episode_reward[agent_i], true_rewards[agent_i][-1])
                        rnn_states[agent_i] = torch.zeros([get_hidden_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0
                        num_frames = 0
                        for i in range(len(ablation_list)):
                            print(ablation_list[i])
                        ablation_list = []


                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(done):
                    if not cfg.no_render:
                        env.render()
                    time.sleep(0.05)

            #if all(finished_episode):
            if done_count >= 1:
                finished_episode = [False] * env.num_agents
                avg_episode_rewards_str, avg_true_reward_str = '', ''
                for agent_i in range(env.num_agents):
                    avg_rew = np.mean(episode_rewards[agent_i])
                    avg_true_rew = np.mean(true_rewards[agent_i])
                    if not np.isnan(avg_rew):
                        if avg_episode_rewards_str:
                            avg_episode_rewards_str += ', '
                        avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
                    if not np.isnan(avg_true_rew):
                        if avg_true_reward_str:
                            avg_true_reward_str += ', '
                        avg_true_reward_str += f'#{agent_i}: {avg_true_rew:.3f}'

                log.info('Avg episode rewards: %s, true rewards: %s', avg_episode_rewards_str, avg_true_reward_str)
                log.info('Avg episode reward: %.3f, avg true_reward: %.3f', np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]), np.mean([np.mean(true_rewards[i]) for i in range(env.num_agents)]))

                break

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    cfg = parse_args(evaluation=True)

    torchbeast_ckpt_path = '/path/to/model/model.tar'

    image_folder_name = 'images_surround'
    random = False

    #torchbeast_ckpt_path = None
    status, avg_reward = enjoy(cfg=cfg, torchbeast_ckpt_path=torchbeast_ckpt_path, image_folder_name=image_folder_name,
                               random=random)

    # ffmpeg -r 30 -i %04d.png -vcodec mpeg4 -y movie_surround_expert.mp4

    return status


if __name__ == '__main__':
    sys.exit(main())