import copy
import glob
import os
import time
from collections import deque
from gym import spaces
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo import algo, utils
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
from ppo.model import Policy
from ppo.storage import RolloutStorage
import matplotlib.pyplot as plt


def initiliaze_master(envs,args,device):
    master_actor_critic = Policy(
        envs.observation_space.shape,
        spaces.Discrete(3))
    master_actor_critic.to(device)

    master_agent = algo.PPO(
        master_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.master_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    master_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              master_actor_critic.recurrent_hidden_state_size)

    return master_actor_critic, master_agent, master_rollouts
  


def warmup_update(args,agents,master_agent,actor_critics,master_actor_critic,device,envs,rollouts,master_rollouts):
        warmup_updates = 8
        obs = envs.reset()
        
        master_rollouts.obs[0].copy_(obs)
        master_rollouts.to(device)
        
        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(
            args.num_env_steps) // args.num_steps // args.num_processes
    
        for j in range(warmup_updates):

            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)
                
                # utils.update_linear_schedule(
                #     master_agent.optimizer, j, num_updates,
                #     # Agent here for a reason
                #     agent.optimizer.lr if args.algo == "acktr" else args.master_lr)

            master_step = 0
            slave_step = 0
            for step in range(args.num_steps):
                # Sample master action            
                with torch.no_grad():
                    master_value, master_action, master_action_log_prob, master_recurrent_hidden_states = master_actor_critic.act(
                        master_rollouts.obs[step], master_rollouts.recurrent_hidden_states[step],
                        master_rollouts.masks[step])
                master_action=master_action.float()
                # print("Master observation: {}, Master action:{}".format(master_rollouts.obs[step],master_action))
                # print("Master observation: {}, Master action:{}".format((master_rollouts.obs[step]*torch.Tensor(np.sqrt(envs.ob_rms.var))+torch.Tensor(envs.ob_rms.mean)).int(),master_action))

                # print("Observation for slave")
                # print(np.concatenate((obs,master_action.detach().cpu().numpy()),axis=1))
                master_reward=0
                # print(master_action)
                selected_subpolicy = int(np.asscalar(master_action.detach().cpu().numpy()))
                actor_critic = actor_critics[selected_subpolicy]
                for i in range(args.master_horizon):               
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts[selected_subpolicy].obs[slave_step], rollouts[selected_subpolicy].recurrent_hidden_states[slave_step],
                            rollouts[selected_subpolicy].masks[slave_step])

                    # Obser reward and next obs
                    
                    obs, reward, done, infos = envs.step(action)
                    master_reward+=reward
                    slave_step+=1

                
                    # for info in infos:
                    #     if 'episode' in info.keys():
                    #         episode_rewards.append(info['episode']['r'])

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0]
                        for info in infos])
                    if(done.any()):
                        break
                master_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                master_bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                
                # print("Master next state: {}".format(obs))
                
                master_rollouts.insert(obs, master_recurrent_hidden_states, master_action,
                                master_action_log_prob, master_value, master_reward, master_masks, master_bad_masks)
                
                

            with torch.no_grad():                
                master_next_value = master_actor_critic.get_value(
                    master_rollouts.obs[-1], master_rollouts.recurrent_hidden_states[-1],
                    master_rollouts.masks[-1]).detach()


            master_rollouts.compute_returns(master_next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)


            master_value_loss, master_action_loss, master_dist_entropy = master_agent.update(master_rollouts)
            master_rollouts.after_update()
    

def joint_update(args,agents,master_agent,actor_critics,master_actor_critic,device,envs,rollouts,master_rollouts):
        joint_updates = 2
            
        obs = envs.reset()
        # print(obs.shape)
        master_rollouts.obs[0].copy_(obs)
        master_rollouts.to(device)
        master_rollouts.step = 0
        # print(obs.shape)
        # print(torch.cat((torch.Tensor(obs),torch.zeros((obs.shape[0],1))),dim=1).shape)
        # print("Rollout step: {}".format(rollouts.step))
        num_sub_policy=3
        for i in range(num_sub_policy):
            rollouts[i].step = 0

        rollout_start = [True for i in range(num_sub_policy)]

        
        # rollouts.obs[0].copy_(obs)
        # rollouts.to(device)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(
            args.num_env_steps) // args.num_steps // args.num_processes
        stats = {}
        for j in range(joint_updates):
            slave_step = 0
            for step in range(args.num_steps):
                with torch.no_grad():
                    master_value, master_action, master_action_log_prob, master_recurrent_hidden_states = master_actor_critic.act(
                        master_rollouts.obs[step], master_rollouts.recurrent_hidden_states[step],
                        master_rollouts.masks[step],should_print=True)
                master_action=master_action.float()
                selected_subpolicy = int(np.asscalar(master_action.detach().cpu().numpy()))
                actor_critic = actor_critics[selected_subpolicy]
                actions = []
                # print("Observation for slave")
                # print(np.concatenate((obs,master_action.detach().cpu().numpy()),axis=1))
                master_reward=0
                if(rollout_start[selected_subpolicy]):
                    rollouts[selected_subpolicy].obs[0].copy_(torch.cat((obs,master_action),dim=1).float())
                    rollouts[selected_subpolicy].to(device)
                rollout_start[selected_subpolicy] = False    
                for i in range(args.master_horizon):               
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts[selected_subpolicy].obs[slave_step], rollouts[selected_subpolicy].recurrent_hidden_states[slave_step],
                            rollouts[selected_subpolicy].masks[slave_step])

                    # Obser reward and next obs
                    obs, reward, done, infos = envs.step(action)
                    # print(" Obs:{} Done: {}".format(obs,done))
                    actions.append(action)
                    master_reward+=reward
                    slave_step+=1
                    # print(obs, reward, done, infos)
                    if(j>joint_updates/2):
                        for info in infos:
                            if 'episode' in info.keys():
                                episode_rewards.append(info['episode']['r'])

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0]
                        for info in infos])
                    rollouts[selected_subpolicy].insert(torch.cat((obs,master_action),dim=1).float(), recurrent_hidden_states, action,
                                    action_log_prob, value, reward, masks, bad_masks)
                    if(done.any()):
                        break
                master_masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                master_bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                # print(obs)
                master_rollouts.insert(obs, master_recurrent_hidden_states, master_action,
                                master_action_log_prob, master_value, master_reward, master_masks, master_bad_masks)
                
            
            next_values = []
            with torch.no_grad():
                # print(rollouts.obs)
                # print(rollouts.masks)
                for sub in range(num_sub_policy):
                    next_values.append(actor_critic.get_value(
                        rollouts[sub].obs[-1], rollouts[sub].recurrent_hidden_states[-1],
                        rollouts[sub].masks[-1]).detach())
                master_next_value = master_actor_critic.get_value(
                    master_rollouts.obs[-1], master_rollouts.recurrent_hidden_states[-1],
                    master_rollouts.masks[-1]).detach()

            for sub in range(num_sub_policy):
                rollouts[sub].compute_returns(next_values[sub], args.use_gae, args.gamma,
                                        args.gae_lambda, args.use_proper_time_limits)

                value_loss, action_loss, dist_entropy = agents[sub].update(rollouts[sub])
                rollouts[sub].after_update()


            master_rollouts.compute_returns(master_next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)


            master_value_loss, master_action_loss, master_dist_entropy = master_agent.update(master_rollouts)



            master_rollouts.after_update()



def evaluate(args,agents,master_agent,actor_critics,master_actor_critic,device,envs,rollouts,master_rollouts,log_data):
    
    # rollouts.obs[0].copy_(obs)
    # rollouts.to(device)
    eval_episodes = 4

    episode_rewards = deque(maxlen=eval_episodes*100)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    stats = {}
    
    for j in range(eval_episodes):
        # print("Evaluating")
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                            args.gamma, args.log_dir, device, False)

        master_actor_critic,master_agent,master_rollouts=initiliaze_master(envs,args,device)

        # Train master only
        warmup_update(args,agents,master_agent,actor_critics,master_actor_critic,device,envs,rollouts,master_rollouts)
        
        
        obs = envs.reset()
        # print(obs.shape)
        master_rollouts.obs[0].copy_(obs)
        master_rollouts.to(device)
        master_rollouts.step = 0
        # print(obs.shape)
        # print(torch.cat((torch.Tensor(obs),torch.zeros((obs.shape[0],1))),dim=1).shape)
        num_sub_policy=3
        for i in range(num_sub_policy):
            rollouts[i].step = 0

        rollout_start = [True for i in range(num_sub_policy)]


        slave_step = 0
        for step in range(args.num_steps):
            with torch.no_grad():
                master_value, master_action, master_action_log_prob, master_recurrent_hidden_states = master_actor_critic.act(
                    master_rollouts.obs[step], master_rollouts.recurrent_hidden_states[step],
                    master_rollouts.masks[step],should_print=True,deterministic=True)
            master_action=master_action.float()
            selected_subpolicy = int(np.asscalar(master_action.detach().cpu().numpy()))
            actor_critic = actor_critics[selected_subpolicy]

            actions = []
            # print("Observation for slave")
            # print(np.concatenate((obs,master_action.detach().cpu().numpy()),axis=1))
            master_reward=0
            # print(rollout_start)
            if(rollout_start[selected_subpolicy]):
                rollouts[selected_subpolicy].obs[0].copy_(torch.cat((obs,master_action),dim=1).float())
                rollouts[selected_subpolicy].to(device)
            rollout_start[selected_subpolicy] = False    
            for i in range(args.master_horizon):               
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts[selected_subpolicy].obs[slave_step], rollouts[selected_subpolicy].recurrent_hidden_states[slave_step],
                        rollouts[selected_subpolicy].masks[slave_step],deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)
                # print(" Obs:{} Done: {}".format(obs,done))
                actions.append(action)
                master_reward+=reward
                slave_step+=1
                # print(obs, reward, done, infos)
                for info in infos:
                    if 'episode' in info.keys():
                        # print("Eval reward:{}, done:{} ".format(info['episode']['r'],done))
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                rollouts[selected_subpolicy].insert(torch.cat((obs,master_action),dim=1).float(), recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)
                if(done.any()):
                    break
            master_masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            master_bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            # print(obs)
            master_rollouts.insert(obs, master_recurrent_hidden_states, master_action,
                            master_action_log_prob, master_value, master_reward, master_masks, master_bad_masks)
            if True:
                # print(actions)
                if(repr(actions) not in stats.keys()):
                    stats[repr(actions)] = [torch.cat((master_rollouts.obs[step],master_action),dim=1)]
                else:
                    found = False
                    for val in stats[repr(actions)]:
                        if(torch.all(torch.eq(val,torch.cat((master_rollouts.obs[step],master_action),dim=1)))):
                        # if(torch.all(torch.eq(val,torch.cat((master_rollouts.obs[step]*torch.Tensor(np.sqrt(envs.ob_rms.var))+torch.Tensor(envs.ob_rms.mean),master_action),dim=1)))):

                            found=True
                            break
                    if not found:
                        stats[repr(actions)].append(torch.cat((master_rollouts.obs[step],master_action),dim=1))
    if len(episode_rewards) > 1:
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        end = time.time()
        print(
            "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            .format(j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards)))
        log_data['mean'].append(np.mean(episode_rewards))
        log_data['median'].append(np.median(episode_rewards))
        log_data['min'].append(np.min(episode_rewards))
        log_data['max'].append(np.max(episode_rewards))
        
        
        
        
    print("Option Discovery Statistics:****************************")
    print("-------------------------------------------------------")
    for key in stats.keys():
        print(key,len(stats[key]))
    print("******************************************************")            
        

def plot_and_save(log_data,log_dir,title="Option learning curve"):

    data = log_data
    y1 = data['mean']
    y2 = data['max']

    x = data['iter']
    # print(x)
    
    #y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y1):]
    x = x[len(x) - len(y2):]


    fig = plt.figure(title)

    plt.plot(x, y1,label="mean")
    plt.plot(x, y2,label="max")
    
    plt.xlabel('Iters')
    plt.ylabel('Evaluation')
    plt.title(title)
    plt.legend(loc = "upper left")
    plt.savefig(log_dir+"learning_curve.png")
    plt.clf()               
    


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Define an environment
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    # Create a network for the policy [MLP]
    master_actor_critic = Policy(
        envs.observation_space.shape,
        spaces.Discrete(8))
    master_actor_critic.to(device)
    
    num_sub_policy = 3
    actor_critics = [Policy(
        envs.observation_space.shape+np.array([1,]),
        envs.action_space) for i in range(num_sub_policy)]
    for actor_critic in actor_critics:
        actor_critic.to(device)
    
    

    # if args.algo == 'ppo':
        # Initialize an agent with the policy network
    master_agent = algo.PPO(
        master_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.master_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    agents = [algo.PPO(
        actor_critics[i],
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm) for i in range(num_sub_policy) ]
        # agent = algo.PPO(
        #     actor_critic,
        #     args.clip_param,
        #     args.ppo_epoch,
        #     args.num_mini_batch,
        #     args.value_loss_coef,
        #     args.entropy_coef,
        #     lr=args.lr,
        #     eps=args.eps,
        #     max_grad_norm=args.max_grad_norm)



    master_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              master_actor_critic.recurrent_hidden_state_size)

    rollouts_sub = [RolloutStorage(args.num_steps*3, args.num_processes,
                              envs.observation_space.shape+np.array([1,]), envs.action_space,
                              actor_critics[i].recurrent_hidden_state_size) for i in range(num_sub_policy)]
    # rollouts = RolloutStorage(args.num_steps*3, args.num_processes,
    #                           envs.observation_space.shape+np.array([1,]), envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size)


    iters = 100
    eval_interval = 10
    print("****************************")
    print("Starting option Discovery")
    print("****************************")
    log_data = {'iter':[],'mean':[],'median':[],'min':[],'max':[]}
    seed = 500
    log_dir = "logs/opt_dump_"+str(seed)+ "/"
    os.makedirs(log_dir, exist_ok=True)
    for i in range(iters):
        
        

        # Start a new environment[Environment selects a random task]
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                            args.gamma, args.log_dir, device, False)
                
        # Initialize master
        master_actor_critic,master_agent,master_rollouts=initiliaze_master(envs,args,device)
        
        # Train master only
        warmup_update(args,agents,master_agent,actor_critics,master_actor_critic,device,envs,rollouts_sub,master_rollouts)
        # print("Num updates :{}".format(num_updates))

        

        # Do joint updates
        joint_update(args,agents,master_agent,actor_critics,master_actor_critic,device,envs,rollouts_sub,master_rollouts)

        if((i+1)%eval_interval==0):
            print("Iteration: {}".format(i))
            log_data['iter'].append(i)
            master_actor_critic,master_agent,master_rollouts=initiliaze_master(envs,args,device)
            evaluate(args,agents,master_agent,actor_critics,master_actor_critic,device,envs,rollouts_sub,master_rollouts,log_data)
            plot_and_save(log_data,log_dir)
    
    np.save(log_dir+"log_data.npy",log_data)    
            

if __name__ == "__main__":
    main()
