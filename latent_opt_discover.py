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

def initiliaze_master(envs,args,device):
    master_actor_critic = Policy(
        envs.observation_space.shape,
        spaces.Discrete(8))
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
  


def warmup_update(args,agent,master_agent,actor_critic,master_actor_critic,device,envs,rollouts,master_rollouts):
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
                
                for i in range(args.master_horizon):               
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[slave_step], rollouts.recurrent_hidden_states[slave_step],
                            rollouts.masks[slave_step])

                    # Obser reward and next obs
                    obs, reward, done, infos = envs.step(action)
                    master_reward+=reward
                    slave_step+=1

                    # print(obs, reward, done, infos)
                
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
    

def joint_update(args,agent,master_agent,actor_critic,master_actor_critic,device,envs,rollouts,master_rollouts):
        joint_updates = 2
            
        obs = envs.reset()
        # print(obs.shape)
        master_rollouts.obs[0].copy_(obs)
        master_rollouts.to(device)
        master_rollouts.step = 0
        # print(obs.shape)
        # print(torch.cat((torch.Tensor(obs),torch.zeros((obs.shape[0],1))),dim=1).shape)
        rollouts.step = 0
        # print("Rollout step: {}".format(rollouts.step))
        rollout_start = True

        
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
                actions = []
                # print("Observation for slave")
                # print(np.concatenate((obs,master_action.detach().cpu().numpy()),axis=1))
                master_reward=0
                if(rollout_start):
                    rollouts.obs[0].copy_(torch.cat((obs,master_action),dim=1).float())
                    rollouts.to(device)
                rollout_start = False    
                for i in range(args.master_horizon):               
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[slave_step], rollouts.recurrent_hidden_states[slave_step],
                            rollouts.masks[slave_step])

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
                    rollouts.insert(torch.cat((obs,master_action),dim=1).float(), recurrent_hidden_states, action,
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
                
            

            with torch.no_grad():
                # print(rollouts.obs)
                # print(rollouts.masks)
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()
                master_next_value = master_actor_critic.get_value(
                    master_rollouts.obs[-1], master_rollouts.recurrent_hidden_states[-1],
                    master_rollouts.masks[-1]).detach()


            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

            master_rollouts.compute_returns(master_next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)


            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            master_value_loss, master_action_loss, master_dist_entropy = master_agent.update(master_rollouts)



            rollouts.after_update()
            master_rollouts.after_update()



def evaluate(args,agent,master_agent,actor_critic,master_actor_critic,device,envs,rollouts,master_rollouts):
    
    # rollouts.obs[0].copy_(obs)
    # rollouts.to(device)
    eval_episodes = 4

    episode_rewards = deque(maxlen=eval_episodes*16)

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
        warmup_update(args,agent,master_agent,actor_critic,master_actor_critic,device,envs,rollouts,master_rollouts)
        
        
        obs = envs.reset()
        # print(obs.shape)
        master_rollouts.obs[0].copy_(obs)
        master_rollouts.to(device)
        master_rollouts.step = 0
        # print(obs.shape)
        # print(torch.cat((torch.Tensor(obs),torch.zeros((obs.shape[0],1))),dim=1).shape)
        rollouts.step = 0
        # print("Rollout step: {}".format(rollouts.step))
        rollout_start = True


        slave_step = 0
        for step in range(args.num_steps):
            with torch.no_grad():
                master_value, master_action, master_action_log_prob, master_recurrent_hidden_states = master_actor_critic.act(
                    master_rollouts.obs[step], master_rollouts.recurrent_hidden_states[step],
                    master_rollouts.masks[step],should_print=True,deterministic=True)
            master_action=master_action.float()
            actions = []
            # print("Observation for slave")
            # print(np.concatenate((obs,master_action.detach().cpu().numpy()),axis=1))
            master_reward=0
            if(rollout_start):
                rollouts.obs[0].copy_(torch.cat((obs,master_action),dim=1).float())
                rollouts.to(device)
            rollout_start = False    
            for i in range(args.master_horizon):               
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[slave_step], rollouts.recurrent_hidden_states[slave_step],
                        rollouts.masks[slave_step],deterministic=True)

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
                rollouts.insert(torch.cat((obs,master_action),dim=1).float(), recurrent_hidden_states, action,
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

    print("Option Discovery Statistics:****************************")
    print("-------------------------------------------------------")
    for key in stats.keys():
        print(key,len(stats[key]))
    print("******************************************************")            
        

            
    
    





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
    

    actor_critic = Policy(
        envs.observation_space.shape+np.array([1,]),
        envs.action_space)

    actor_critic.to(device)
    
    

    if args.algo == 'ppo':
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

        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)



    master_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              master_actor_critic.recurrent_hidden_state_size)

    rollouts = RolloutStorage(args.num_steps*3, args.num_processes,
                              envs.observation_space.shape+np.array([1,]), envs.action_space,
                              actor_critic.recurrent_hidden_state_size)


    iters = 100000
    eval_interval = 10
    
    for i in range(iters):
        
        print("Iteration: {}".format(i))

        # Start a new environment[Environment selects a random task]
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                            args.gamma, args.log_dir, device, False)
                
        # Initialize master
        master_actor_critic,master_agent,master_rollouts=initiliaze_master(envs,args,device)
        
        # Train master only
        warmup_update(args,agent,master_agent,actor_critic,master_actor_critic,device,envs,rollouts,master_rollouts)
        # print("Num updates :{}".format(num_updates))

        

        # Do joint updates
        joint_update(args,agent,master_agent,actor_critic,master_actor_critic,device,envs,rollouts,master_rollouts)
        # joint_updates = 2
        if((i+1)%eval_interval==0):
             master_actor_critic,master_agent,master_rollouts=initiliaze_master(envs,args,device)
             evaluate(args,agent,master_agent,actor_critic,master_actor_critic,device,envs,rollouts,master_rollouts)
            
        # obs = envs.reset()
        # # print(obs.shape)
        # master_rollouts.obs[0].copy_(obs)
        # master_rollouts.to(device)
        # master_rollouts.step = 0
        # # print(obs.shape)
        # # print(torch.cat((torch.Tensor(obs),torch.zeros((obs.shape[0],1))),dim=1).shape)
        # rollouts.step = 0
        # # print("Rollout step: {}".format(rollouts.step))
        # rollout_start = True

        
        # # rollouts.obs[0].copy_(obs)
        # # rollouts.to(device)

        # episode_rewards = deque(maxlen=10)

        # start = time.time()
        # num_updates = int(
        #     args.num_env_steps) // args.num_steps // args.num_processes
        # stats = {}
        # # print("Num updates :{}".format(num_updates))
        # for j in range(joint_updates):

        #     # if args.use_linear_lr_decay:
        #     #     # decrease learning rate linearly
        #     #     utils.update_linear_schedule(
        #     #         agent.optimizer, j, num_updates,
        #     #         agent.optimizer.lr if args.algo == "acktr" else args.lr)
        #     #     utils.update_linear_schedule(
        #     #         master_agent.optimizer, j, num_updates,
        #     #         # Agent here for a reason
        #     #         agent.optimizer.lr if args.algo == "acktr" else args.master_lr)
        #     master_step = 0
        #     slave_step = 0
        #     for step in range(args.num_steps):
        #         # Sample master action
        #         # print(envs.venv.task)
        #         with torch.no_grad():
        #             master_value, master_action, master_action_log_prob, master_recurrent_hidden_states = master_actor_critic.act(
        #                 master_rollouts.obs[step], master_rollouts.recurrent_hidden_states[step],
        #                 master_rollouts.masks[step],should_print=True)
        #         master_action=master_action.float()
        #         actions = []
        #         # print("Observation for slave")
        #         # print(np.concatenate((obs,master_action.detach().cpu().numpy()),axis=1))
        #         master_reward=0
        #         if(rollout_start):
        #             rollouts.obs[0].copy_(torch.cat((obs,master_action),dim=1).float())
        #             rollouts.to(device)
        #         rollout_start = False    
        #         for i in range(args.master_horizon):               
        #             # Sample actions
        #             with torch.no_grad():
        #                 value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
        #                     rollouts.obs[slave_step], rollouts.recurrent_hidden_states[slave_step],
        #                     rollouts.masks[slave_step])

        #             # Obser reward and next obs
        #             obs, reward, done, infos = envs.step(action)
        #             # print(" Obs:{} Done: {}".format(obs,done))
        #             actions.append(action)
        #             master_reward+=reward
        #             slave_step+=1
        #             # print(obs, reward, done, infos)
        #             if(j>joint_updates/2):
        #                 for info in infos:
        #                     if 'episode' in info.keys():
        #                         episode_rewards.append(info['episode']['r'])

        #             # If done then clean the history of observations.
        #             masks = torch.FloatTensor(
        #                 [[0.0] if done_ else [1.0] for done_ in done])
        #             bad_masks = torch.FloatTensor(
        #                 [[0.0] if 'bad_transition' in info.keys() else [1.0]
        #                 for info in infos])
        #             rollouts.insert(torch.cat((obs,master_action),dim=1).float(), recurrent_hidden_states, action,
        #                             action_log_prob, value, reward, masks, bad_masks)
        #             if(done.any()):
        #                 break
                    
        #         if True:
        #         # if(j>joint_updates/2):
        #             if(repr(actions) not in stats.keys()):
        #                 stats[repr(actions)] = [torch.cat((master_rollouts.obs[step],master_action),dim=1)]
        #             else:
        #                 found = False
        #                 for val in stats[repr(actions)]:
        #                     if(torch.all(torch.eq(val,torch.cat((master_rollouts.obs[step],master_action),dim=1)))):
        #                     # if(torch.all(torch.eq(val,torch.cat((master_rollouts.obs[step]*torch.Tensor(np.sqrt(envs.ob_rms.var))+torch.Tensor(envs.ob_rms.mean),master_action),dim=1)))):

        #                         found=True
        #                         break
        #                 if not found:
        #                     stats[repr(actions)].append(torch.cat((master_rollouts.obs[step],master_action),dim=1))

        #                     # stats[repr(actions)].append(torch.cat((master_rollouts.obs[step]*torch.Tensor(np.sqrt(envs.ob_rms.var))+torch.Tensor(envs.ob_rms.mean),master_action),dim=1))
        #             # if(torch.cat((master_rollouts.obs[step],master_action),dim=1) not in stats[repr(actions)]):
        #             #     stats[repr(actions)].append(torch.cat((master_rollouts.obs[step],master_action),dim=1))    
        #         # stats[repr(actions)]
        #         # print(actions)
                
                
        #         master_masks = torch.FloatTensor(
        #             [[0.0] if done_ else [1.0] for done_ in done])
        #         master_bad_masks = torch.FloatTensor(
        #             [[0.0] if 'bad_transition' in info.keys() else [1.0]
        #             for info in infos])
        #         # print(obs)
        #         master_rollouts.insert(obs, master_recurrent_hidden_states, master_action,
        #                         master_action_log_prob, master_value, master_reward, master_masks, master_bad_masks)
                
            

        #     with torch.no_grad():
        #         # print(rollouts.obs)
        #         # print(rollouts.masks)
        #         next_value = actor_critic.get_value(
        #             rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
        #             rollouts.masks[-1]).detach()
        #         master_next_value = master_actor_critic.get_value(
        #             master_rollouts.obs[-1], master_rollouts.recurrent_hidden_states[-1],
        #             master_rollouts.masks[-1]).detach()


        #     rollouts.compute_returns(next_value, args.use_gae, args.gamma,
        #                             args.gae_lambda, args.use_proper_time_limits)

        #     master_rollouts.compute_returns(master_next_value, args.use_gae, args.gamma,
        #                             args.gae_lambda, args.use_proper_time_limits)


        #     value_loss, action_loss, dist_entropy = agent.update(rollouts)
        #     master_value_loss, master_action_loss, master_dist_entropy = master_agent.update(master_rollouts)



        #     rollouts.after_update()
        #     master_rollouts.after_update()

        #     # save for every interval-th episode or for the last epoch
        #     if (j % args.save_interval == 0
        #             or j == num_updates - 1) and args.save_dir != "":
        #         save_path = os.path.join(args.save_dir, args.algo)
        #         try:
        #             os.makedirs(save_path)
        #         except OSError:
        #             pass

        #         torch.save([
        #             actor_critic,
        #             getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #         ], os.path.join(save_path, args.env_name + ".pt"))

        #     # print("Logging")
        #     # print(episode_rewards)
        #     if j % args.log_interval == 0 and len(episode_rewards) > 1:
        #         total_num_steps = (j + 1) * args.num_processes * args.num_steps
        #         end = time.time()
        #         print(
        #             "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
        #             .format(j, total_num_steps,
        #                     int(total_num_steps / (end - start)),
        #                     len(episode_rewards), np.mean(episode_rewards),
        #                     np.median(episode_rewards), np.min(episode_rewards),
        #                     np.max(episode_rewards), dist_entropy, value_loss,
        #                     action_loss))

        #     if (args.eval_interval is not None and len(episode_rewards) > 1
        #             and j % args.eval_interval == 0):
        #         ob_rms = utils.get_vec_normalize(envs).ob_rms
        #         evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #                 args.num_processes, eval_log_dir, device)



        # # if i%eval_interval == 0:
            


        # print("Option Discovery Statistics:****************************")
        # # for s1 in [1,2]:
        # #     for s2 in [0,1,2]:
        # #         for macro in range(8):
        # #             obs = np.array([s1,s2,macro])
        # #             obs_norm = np.clip((obs - self.ob_rms.mean) /
        # #                   np.sqrt(self.ob_rms.var + self.epsilon),
        # #                   -self.clipob, self.clipob)
                    

                    
        # # print(stats)
        # print("-------------------------------------------------------")
        # for key in stats.keys():
        #     print(key,len(stats[key]))
        # print("******************************************************")

if __name__ == "__main__":
    main()
