

def fanin_init(size, fanin=None):
    """
    weight initializer known from https://arxiv.org/abs/1502.01852
    :param size:
    :param fanin:
    :return:
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1 + action_dim, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        return critic Q(s,a)
        :param state: state [n, state_dim] (n is batch_size)
        :param action: action [n, action_dim]
        :return: Q(s,a) [n, 1]
        """

        s1 = self.relu(self.fc1(state))
        x = torch.cat((s1, action), dim=1)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        """
        :param state_dim: int
        :param action_dim: int
        :param action_lim: Used to limit action space in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, action_dim)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        return actor policy function Pi(s)
        :param state: state [n, state_dim]
        :return: action [n, action_dim]
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x)) # tanh limit (-1, 1)
        return action


class DDPG:
    def __init__(self, config: Config):
        self.config = config
        self.init()

    def init(self):
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.is_training = True
        self.randomer = OUNoise(self.action_dim)
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.config.learning_rate_actor)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.config.learning_rate)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        if self.config.use_cuda:
            self.cuda()

    def learning(self):
        s1, a1, r1, t1, s2 = self.buffer.sample_batch(self.batch_size)
        # bool -> int
        t1 = (t1 == False) * 1
        s1 = torch.tensor(s1, dtype=torch.float)
        a1 = torch.tensor(a1, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float)
        t1 = torch.tensor(t1, dtype=torch.float)
        s2 = torch.tensor(s2, dtype=torch.float)
        if self.config.use_cuda:
            s1 = s1.cuda()
            a1 = a1.cuda()
            r1 = r1.cuda()
            t1 = t1.cuda()
            s2 = s2.cuda()

        a2 = self.actor_target(s2).detach()
        target_q = self.critic_target(s2, a2).detach()
        y_expected = r1[:, None] + t1[:, None] * self.config.gamma * target_q
        y_predicted = self.critic.forward(s1, a1)

        # critic gradient
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # actor gradient
        pred_a = self.actor.forward(s1)
        loss_actor = (-self.critic.forward(s1, pred_a)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Notice that we only have gradient updates for actor and critic, not target
        # actor_optimizer.step() and critic_optimizer.step()

        soft_update(self.actor_target, self.actor, self.config.tau)
        soft_update(self.critic_target, self.critic, self.config.tau)

        return loss_actor.item(), loss_critic.item()


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def decay_epsilon(self):
        self.epsilon -= self.config.eps_decay

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        if self.config.use_cuda:
            state = state.cuda()

        action = self.actor(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action += self.is_training * max(self.epsilon, self.config.epsilon_min) * self.randomer.noise()
        action = np.clip(action, -1.0, 1.0)

        self.action = action
        return action

    def reset(self):
        self.randomer.reset()

    def load_weights(self, output):
        if output is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def save_config(self, output, save_obj=False):

        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

        if save_obj:
            file = open(output + '/config.obj', 'wb')
            pickle.dump(self.config, file)
            file.close()

    def save_checkpoint(self, ep, total_step, output):

        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)

        torch.save({
            'episodes': ep,
            'total_step': total_step,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, '%s/checkpoint_ep_%d.tar'% (checkpath, ep))


    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        episode = checkpoint['episodes']
        total_step = checkpoint['total_step']
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        return episode, total_step
