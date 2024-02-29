import argparse
from isaacgymenvs.utils.pbrl.mpi import str2bool, str2list


def argparser():
    parser = argparse.ArgumentParser('PBRL Environment',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # PBRL
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--pbrl', type=str2bool, default=True, help="Enable PBRL")
    parser.add_argument('--num_agents', type=int, default=4, help="number of agent(s)")
    parser.add_argument('--start_evo', type=int, default=10e6, help="initial training steps before evolution")
    parser.add_argument('--evo_interval', type=int, default=2e6, help="how frequently to do evolution")
    parser.add_argument('--pbrl_params', type=str, default='ppo', 
                        help="name of json file with hyperparams to tune (without '.json' extension)")
    parser.add_argument('--mut_scheme', type=str, default='perturb', choices=['perturb', 'sample', 'dex-pbt'],
                        help="which mutation scheme to use")

    # RL Algorithm
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['ppo', 'sac', 'ddpg'])

    # Vanilla RL
    parser.add_argument('--hid_size', type=list, default=[512, 256, 128])
    parser.add_argument('--activation', type=str, default='elu',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--lr', type=float, default=5e-4, help='the initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')

    # PPO
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--value_loss_coeff', type=float, default=1)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--adv_norm', type=str2bool, default=False)
    parser.add_argument('--value_norm', type=str2bool, default=False)
    parser.add_argument('--value_bootstrap', type=str2bool, default=False)
    parser.add_argument('--lr_schedule', type=str, default='adaptive',
                         choices=['linear', 'adaptive', 'fixed'])

    # SAC & DDPG
    parser.add_argument('--nstep', type=int, default=3)
    parser.add_argument('--memory_size', type=int, default=int(1e6))

    # Training
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--ob_norm', type=str2bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=8, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='the sample batch size')
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--max_global_step', type=int, default=int(200e6), help='total no. of steps for training')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task', type=str, default='AnymalTerrain')
    parser.add_argument('--sub_task', type=str, default='throw', help='only for AllegroKuka tasks')
    parser.add_argument('--headless', type=str2bool, default=False)

    # Log
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--evaluate_interval', type=int, default=10)
    parser.add_argument('--ckpt_interval', type=int, default=100)
    parser.add_argument('--log_root_dir', type=str, default='log')
    parser.add_argument('--wandb', type=str2bool, default=False,
                        help='set it True if you want to use wandb')
    parser.add_argument('--wandb_entity', type=str, default='panda', help='wandb user name')
    parser.add_argument('--wandb_project', type=str, default='anymal-terrain')

    # Evaluation
    parser.add_argument('--sim', type=str2bool, default=True,
                        help='whether to test the policy in simulation or real robot')
    parser.add_argument('--ckpt_num', type=int, default=None)
    parser.add_argument('--num_eval', type=int, default=20)
    parser.add_argument('--num_record_samples', type=int, default=1,
                        help='number of trajectories to collect during eval')

    # Misc
    parser.add_argument('--prefix', type=str, default='4.anymal')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--seed', type=int, default=40, help='random seed')

    args, unparsed = parser.parse_known_args()

    return args, unparsed
