import argparse
import copy

import AOP_project.params.default_params as default_params
import AOP_project.params.env_params as env_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_plant', '-p', type=int, default=1,
        help='# of plants to use for environment')
    parser.add_argument('--env', '-e', type=str, default='pendulum',
        choices=['pendulum, hopper', 'ant', 'maze-d', 'maze-s'],
        help='Base environment for agent')
    parser.add_argument('--algo', '-a', type=str, default='polo',
        choices=['aop', 'mpc-3', 'polo', 'td3'],
        help='Choice of algorithm to use for training')
    parser.add_argument('--setting', '-s', type=str, default='changing',
        choices=['changing', 'novel', 'standard'],
        help='Specify which setting to test in')
    parser.add_argument('--output_dir', '-d', type=str,
        help='Directory in ex/ to output models to (for example, ex/my_exp_1)')
    parser.add_argument('--num_trials', '-n', type=int, default=1,
        help='Number of trials (seeds) to run for')
    parser.add_argument('--num_cpus', '-c', type=int, default=1,
        help='Number of CPUs to use for trajectory generation')
    parser.add_argument('--use_gpu', '-g', default=True,
        help='Whether or not to use GPU (currently only TD3 supports this)')
    parser.add_argument('--test_pol', '-t', default=True,
        help='Whether or not to test the policy in standard episode')
    
    args = parser.parse_args()

    # Basic information for experiments
    agent_class = get_agent_class(args.algo)
    output_dir = args.output_dir if args.output_dir else default_output_dir()

    # Setting parameter settings for experiments
    params = copy.deepcopy(default_params.base_params)
    params.update(env_params.env_params[args.env][args.setting])
    params['problem']['algo'] = args.algo
    params['problem']['output_dir'] = output_dir

    params['mpc']['num_cpu'] = args.num_cpus
    params['pg']['num_cpu'] = args.num_cpus

    params['problem']['test_pol'] = args.test_pol
    params['problem']['eval_len'] = 1000
    params['problem']['use_gpu'] = args.use_gpu

    # Update parameter setting for inverted pendulum
    pend_conf = {}
    amplitude_list = [0.1, 0.15, 0.2, 0.2, 0.2]
    frequency_list = [0.01, 0.15, 0.2, 0.2, 0.2]
    trigger_list = [10, 10, 10, 10, 10]  # ms
    for i in range(args.num_plant):
        pend_conf['pend_%s'%(i)] = {'id': i,
                                    'amplitude': amplitude_list[i],
                                    'frequency': frequency_list[i],
                                    'trigger_time': trigger_list[i]}
    params.update(pend_conf)

    params_ = {
        'state_dim' : 5*args.num_plant,
        'action_dim': 1,
        'min_action': -0.5,
        'max_action': 0.5
    }
    params.update(params_)
    tmp = {'num_plant':args.num_plant}
    params.update(tmp)
    

    # Setting algorithm-specific hyperparameter settings
    if args.algo == 'polo' or args.algo == 'mpc-3':
        params['mpc']['num_iter'] = 4

    # Run experiments
    for i in range(args.num_trials):
        params['problem']['dir_name'] = '%s/trial_%d' % (output_dir, i)
        agent = agent_class(params)
        agent.run_lifetime()

def default_output_dir():
    import datetime
    now = datetime.datetime.now()
    ctime = '%02d%02d_%02d%02d' % (now.month, now.day, now.hour, now.minute)
    return 'ex/' + ctime

def get_agent_class(algo):
    if algo == 'mpc-8' or algo == 'mpc-3':
        from AOP_project.agents.MPCAgent import MPCAgent
        agent_class = MPCAgent
    elif algo == 'polo':
        from AOP_project.agents.POLOAgent import POLOAgent
        agent_class = POLOAgent
    elif algo == 'aop':
        from AOP_project.agents.AOPTD3Agent import AOPTD3Agent
        agent_class = AOPTD3Agent
    elif algo == 'td3':
        from AOP_project.agents.TD3Agent import TD3Agent
        agent_class = TD3Agent
    
    return agent_class


if __name__ == '__main__':
    main()