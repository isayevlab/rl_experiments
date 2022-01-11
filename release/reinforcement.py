"""
This class implements simple policy gradient algorithm for
biasing the generation of molecules towards desired values of
properties aka Reinforcement Learninf for Structural Evolution (ReLeaSE)
as described in 
Popova, M., Isayev, O., & Tropsha, A. (2018). 
Deep reinforcement learning for de novo drug design. 
Science advances, 4(7), eaap7885.
"""

import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem


class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward):
        """
        Constructor for the Reinforcement object.

        Parameters
        ----------
        generator: object of type StackAugmentedRNN
            generative model that produces string of characters (trajectories)

        predictor: object of any predictive model type
            predictor accepts a trajectory and returns a numerical
            prediction of desired property for the given trajectory

        get_reward: function
            custom reward function that accepts a trajectory, predictor and
            any number of positional arguments and returns a single value of
            the reward for the given trajectory
            Example:
            reward = get_reward(trajectory=my_traj, predictor=my_predictor,
                                custom_parameter=0.97)

        Returns
        -------
        object of type Reinforcement used for biasing the properties estimated
        by the predictor of trajectories produced by the generator to maximize
        the custom reward function get_reward.
        """

        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward

    def policy_gradient(self, data, n_batch=10, gamma=0.97,
                        std_smiles=False, grad_clipping=None, 
                        replay=False, replay_data=None, **kwargs):
        """
        Implementation of the policy gradient algorithm.

        Parameters:
        -----------

        data: object of type GeneratorData
            stores information about the generator data format such alphabet, etc

        n_batch: int (default 10)
            number of trajectories to sample per batch. When training on GPU
            setting this parameter to to some relatively big numbers can result
            in out of memory error. If you encountered such an error, reduce
            n_batch.

        gamma: float (default 0.97)
            factor by which rewards will be discounted within one trajectory.
            Usually this number will be somewhat close to 1.0.


        std_smiles: bool (default False)
            boolean parameter defining whether the generated trajectories will
            be converted to standardized SMILES before running policy gradient.
            Leave this parameter to the default value if your trajectories are
            not SMILES.

        grad_clipping: float (default None)
            value of the maximum norm of the gradients. If not specified,
            the gradients will not be clipped.

        kwargs: any number of other positional arguments required by the
            get_reward function.

        Returns
        -------
        total_reward: float
            value of the reward averaged through n_batch sampled trajectories

        rl_loss: float
            value for the policy_gradient loss averaged through n_batch sampled
            trajectories

        """
        rl_loss = 0
        self.generator.optimizer.zero_grad()
        total_reward = 0
        
        if replay and replay_data is None:
            raise ValueError
        
        for _ in range(n_batch):
            
            if replay:
                idx = np.random.randint(low=0, high=len(replay_data))
                trajectory = replay_data[idx]
                reward = self.get_reward(trajectory, self.predictor, **kwargs)
                trajectory = '<' + trajectory + '>'
            else:  
                # Sampling new trajectory
                reward = 0
                trajectory = '<>'
                while reward == 0:
                    trajectory = self.generator.evaluate(data, batch_size=1)
                    if len(trajectory) == 1:
                        trajectory = trajectory[0]
                        if std_smiles:
                            try:
                                mol = Chem.MolFromSmiles(trajectory)
                                trajectory = '<' + Chem.MolToSmiles(mol) + '>'
                                reward = self.get_reward(trajectory[1:-1], 
                                                         self.predictor, 
                                                         **kwargs)
                            except:
                                reward = 0
                        else:
                            # added start and end tokens
                            trajectory = '<' + trajectory + '>'
                            reward = self.get_reward(trajectory[1:-1],
                                                     self.predictor, 
                                                     **kwargs)

            # Converting string of characters into tensor
            trajectory_input, _ = data.seq2tensor(trajectory, data.all_characters, flip=False)
            trajectory_input = torch.tensor(trajectory_input, device="cuda", dtype=torch.long)
            discounted_reward = reward
            total_reward += reward

            # Initializing the generator's hidden state
            hidden = self.generator.init_hidden(batch_size=1)
            if self.generator.has_cell:
                cell = self.generator.init_cell(batch_size=1)
                hidden = (hidden, cell)
            if self.generator.has_stack:
                stack = self.generator.init_stack(batch_size=1)
            else:
                stack = None

            # "Following" the trajectory and accumulating the loss
            for p in range(len(trajectory)-1):
                output, hidden, stack = self.generator(trajectory_input[p], 
                                                       hidden, 
                                                       stack)
                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[p+1]
                rl_loss -= (log_probs[0, top_i]*discounted_reward)
                discounted_reward = discounted_reward * gamma

        # Doing backward pass and parameters update
        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch
        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                           grad_clipping)

        self.generator.optimizer.step()
        
        return total_reward, rl_loss.item()
    
    def fine_tune(self, data, n_steps, batch_size, **kwargs):
        _ = self.generator.fit(data=data, n_iterations=n_steps, batch_size=batch_size, **kwargs)
