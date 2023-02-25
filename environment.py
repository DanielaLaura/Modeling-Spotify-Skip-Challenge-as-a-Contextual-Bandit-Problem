import numpy as np

class ContextualEnvironment():
    def __init__(self, model, features, target, batch_size, n_choices):
        self.model = model
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.n_choices = n_choices
        self.compute_initial_rewards()
        self.simulate_rounds_stoch()

    def compute_initial_rewards(self, models):
        rewards_lucb, rewards_aac, rewards_sft, rewards_agr, \
        rewards_ac, rewards_egr = [list() for i in range(len(models))]

        lst_rewards = [rewards_lucb, rewards_aac, rewards_sft,
                       rewards_agr, rewards_ac, rewards_egr]

        # initial seed - all policies start with the same small random selection of actions/rewards
        first_batch = self.features[:self.batch_size, :]
        np.random.seed(1)
        target = np.reshape(self.target, (-1, 1))
        action_chosen = np.random.randint(self.n_choices, size=self.batch_size)
        rewards_received = []
        for action in action_chosen:
            rewards_received.append(int(self.target[action]))
        for model in models:
            model.fit(X=first_batch, a=action_chosen, r=np.array(rewards_received))

    def simulate_rounds_stoch(self, rewards, actions_hist, features_batch, target_batch, rnd_seed):
        np.random.seed(rnd_seed)
        ## choosing actions for this batch
        actions_this_batch = self.model.predict(features_batch).astype('uint8')
        # keeping track of the sum of rewards received
        rewards.append(int(target_batch.sum()))
        # adding this batch to the history of selected actions
        new_actions_hist = np.append(actions_hist, actions_this_batch)
        # rewards obtained now
        rewards_batch = []
        for action in actions_this_batch:
            rewards_batch.append(int(target_batch[action]))
        # now refitting the algorithms after observing these new rewards
        np.random.seed(rnd_seed)
        self.model.partial_fit(features_batch, actions_this_batch, np.array(rewards_batch))

        return new_actions_hist
    def run_simulations(self, batch_size, features, target, models, lst_actions):
        for i in range(int(np.floor(features.shape[0] / batch_size))):
            batch_st = (i + 1) * batch_size
            batch_end = (i + 2) * batch_size
            batch_end = np.min([batch_end, features.shape[0]])

            X_batch = features[batch_st:batch_end, :]
            y_batch = target[batch_st:batch_end, :]
            print(y_batch.sum())
            for model in range(len(models)):
                lst_actions[model] = simulate_rounds_stoch(models[model],
                                                           lst_rewards[model],
                                                           lst_actions[model],
                                                           X_batch, y_batch,
                                                           rnd_seed=batch_st)