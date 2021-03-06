import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner
import tqdm


def constfn(val):
    def f(_):
        return val
    return f

def remap_pong_action(aa):
    if aa in [2, 3]:
        return aa
    if aa in [4, 5]:
        return aa - 2
    else:
        return 0

def train_batch_acc(model, ob_s, a_s, is_pong):
    correct = 0
    for ob, a in zip(ob_s, a_s):
        a_ = model.step(ob)[0]

        # remap action 4, 5 to 2, 3 respectively
        if is_pong:
            a_, a = remap_pong_action(a_), remap_pong_action(a)

        if a_ == a:
            correct += 1
    return (correct / len(a_s))

def pretrain_subset(model,memory_path,index_path,is_pong):
    import pickle
    import numpy as np

    with open(memory_path, 'rb') as f:
        data = pickle.load(f)
    with open(index_path,'rb') as f:
        inds = pickle.load(f)
        np.random.shuffle(inds)

    obs_ = []
    actions_ = []

    returns_ = []
    masks_ = []
    values_ = []
    neglogpacs_ = []
    for obs, returns, masks, actions, values, neglogpacs in data:
        obs_.append(obs)
        returns_.append(returns)
        masks_.append(masks)

        actions_.append(actions)
        values_.append(values)
        neglogpacs_.append(neglogpacs)
    obs_ = np.vstack(obs_)
    returns_ = np.hstack(returns_)
    masks_ = np.hstack(masks_)
    actions_ = np.hstack(actions_)
    values_ = np.hstack(values_)
    neglogpacs_ = np.hstack(neglogpacs_)
    print(obs_.shape,returns_.shape,masks_.shape,actions_.shape,values_.shape,neglogpacs_.shape)

    tot = obs_.shape[0]


    size = inds.shape[0]
    if size == 0:
        print ("no pretraining here")
        return
    print("training with subset of size ",size)

    #np.random.shuffle(inds)

    # nbatch_train = 1280
    nbatch_train = model.nbatch_train
    lrnow = 3e-4
    cliprangenow = 0.2
    noptepochs = int(tot/size)

    train_batch_accs = [0.0]

    def stop_condition(accs):
        assert len(accs) > 2000, "malformed"
        cur = sum(accs[-1000:])
        prev = sum(accs[-2000:-1000])
        # fraction of improvement is small enough, i.e. we stopped improving
        if  (cur - prev) / cur < 0.01:
            return True
        if sum(accs[-100:]) / 100 < 0.9:
            return True
        return False

#    for _ in tqdm.tqdm(range(noptepochs)):
    while True:#sum(train_batch_accs[-100:]) / 100 < 0.9:
        for start in range(0, size, nbatch_train):
            end = start + nbatch_train
            end = min(size,end)
            if end-start == nbatch_train:
                mbinds = inds[start:end]
            else:
                mbinds = inds[end-nbatch_train:end]
            slices = (arr[mbinds] for arr in (obs_, returns_, masks_, actions_, values_, neglogpacs_))
            the_stats = model.train(lrnow, cliprangenow, *slices, supervised=True)
            ob_batch = obs_[mbinds]
            act_batch = actions_[mbinds]
            # track some statistics
            train_batch_accs.append(train_batch_acc(model, ob_batch, act_batch, is_pong))
            # do stats track and early termination every 2000 iterations
            print(len(train_batch_accs))
            if len(train_batch_accs) > 2000:
                print ('past few train_batch_acc was ', sum(train_batch_accs[-1000:]) / 1000)
                print ('stats ', the_stats)
                if stop_condition(train_batch_accs):
                    print ("stop condition reached, pre-training is over")
                    return
                # clear out the accuracies
                train_batch_accs = []

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, mode = 'raw', index_path = None,
          memory_path=None,loss_path=None,is_pong=True,**network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''
    #mode = 'raw' # 'raw', 'memory', 'pretrain'
    #index_path = '../pong_tiers_32.p'  #subset index pickle
    #memory_path = '../ppo2_memory'  # the memory pickle
    #loss_path = 'ppo2_losses_subset_selection_50000'  # the loss pickle which we use to plot the loss curve
    assert mode in ["train_expert", "collect_supervision", "evaluate_pretrain"]
    
    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    if mode == 'evaluate_pretrain':
        print('pretraining with subset of pickle',index_path)
        print('is_pong is ',is_pong)
        
        pretrain_subset(model,memory_path,index_path, is_pong)

    tfirststart = time.time()
    print('total_timesteps',total_timesteps)
    nupdates = total_timesteps//nbatch
    print('n_updates',nupdates)
    if mode == 'collect_supervision':
        print('saving supervision memory path is ',memory_path)
    memory = []
    losses = []
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if mode == 'collect_supervision':
            memory.append((obs,returns,masks,actions,values,neglogpacs))
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    # freeze model during collection
                    if mode != 'collect_supervision':
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    # freeze model during collection
                    if mode != 'collect_supervision':
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            losses.append((update,safemean([epinfo['r'] for epinfo in epinfobuf])))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('time_elapsed', tnow - tfirststart)
            if mode != 'collect_supervision':
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    import pickle
    if mode == 'collect_supervision':
        assert len(memory) > 0, "memory is empty !"
        with open(memory_path, 'wb') as f:
            pickle.dump(memory, f)
            print ("saved expert supervision memory at ", memory_path)
    if loss_path is not None:
        with open(loss_path, 'wb') as f:
            pickle.dump(losses, f)
            print ("loss dumped at ", loss_path)

    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



