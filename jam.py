import numpy as np

def hierarchical_jam(metrics):  # Canonical JAM objective (diagnostic)
    group_scores = np.prod(metrics, axis=1)**(1/5)
    return np.log(np.min(group_scores))

class JAMEnv:
    def __init__(self, coupling=1.2, noise=0.05, budget_per_step=0.2, n_steps=1000,
                 use_geo=True, use_min=True, use_log=True, cost_asym=True, flat_baseline=False):
        self.coupling = coupling
        self.noise = noise
        self.budget = budget_per_step
        self.n_steps = n_steps
        self.use_geo = use_geo
        self.use_min = use_min
        self.use_log = use_log
        self.cost_asym = cost_asym
        self.flat_baseline = flat_baseline
        self.costs = np.array([1,2,3,5,10]) if cost_asym else np.ones(5)
        self.effs = 1.0 / self.costs
        self.metrics = np.ones((6,5))
        self.true_J_history = []
        self.decision_obs_J_history = []
        self.poststep_observed_J_history = []
        self.preclamp_min_sub_history = []
        self.state_min_sub_history = []
        self.true_bottleneck_history = []
        self.decision_obs_bottleneck_history = []
        self.poststep_observed_bottleneck_history = []
        self.energy_cheap = 0.0
        self.energy_exp = 0.0
        self.step = 0
        self.floor_breach_attempts = 0

    def observe(self):
        noisy = self.metrics * (1 + self.noise * np.random.randn(*self.metrics.shape))
        return np.maximum(0.01, noisy)  # Clipped multiplicative observation noise (upward bias near floor)

    def compute_J(self, vals):
        if self.flat_baseline:
            return np.sum(vals)
        group_scores = np.prod(vals, axis=1)**(1/5) if self.use_geo else np.mean(vals, axis=1)
        across = np.min(group_scores) if self.use_min else np.sum(group_scores)
        return np.log(across) if self.use_log else across

    def step_agent(self, energy_alloc):  # energy_alloc: 6x5 budget spent per sub-metric
        # Assertions for allocation validity
        if energy_alloc.shape != (6, 5):
            raise ValueError("energy_alloc must be (6,5)")
        if not np.all(np.isfinite(energy_alloc)):
            raise ValueError("energy_alloc must be finite")
        if not np.all(energy_alloc >= 0):
            raise ValueError("energy_alloc must be nonnegative")
        if not np.isclose(np.sum(energy_alloc), self.budget, atol=1e-6):
            raise ValueError(f"energy_alloc sum must equal budget ({self.budget})")

        alloc = energy_alloc.copy()  # Avoid mutation
        # Coupling in metric delta space (post-coupling alloc may contain negative entries representing environment-imposed adverse deltas, not agent-spent energy)
        delta_g1m1 = alloc[0,0] * self.effs[0]
        delta_g2m1 = -self.coupling * delta_g1m1
        # Adjust energy for G2m1 (eff[0]=1.0)
        alloc[1,0] += delta_g2m1 / self.effs[0]
        # Compute metric deltas
        deltas = alloc * self.effs[None,:]
        # Update with clamp
        pre_clamp = self.metrics + deltas
        self.metrics = np.maximum(0.01, pre_clamp)
        # Diagnostics: breach attempts (pre-clamp <0.01)
        self.floor_breach_attempts += np.sum(pre_clamp < 0.01)
        # True signals
        true_j = self.compute_J(self.metrics)
        self.true_J_history.append(true_j)
        self.preclamp_min_sub_history.append(np.min(pre_clamp))  # Natural dynamics
        self.state_min_sub_history.append(self.metrics.min())  # Realized state
        true_group = np.prod(self.metrics, axis=1)**(1/5) if self.use_geo else np.mean(self.metrics, axis=1)
        true_bottleneck = np.argmin(true_group) if not self.flat_baseline else -1
        self.true_bottleneck_history.append(true_bottleneck)
        # Post-step observed
        post_obs = self.observe()
        post_obs_j = self.compute_J(post_obs)
        self.poststep_observed_J_history.append(post_obs_j)
        post_obs_group = np.prod(post_obs, axis=1)**(1/5) if self.use_geo else np.mean(post_obs, axis=1)
        post_obs_bottleneck = np.argmin(post_obs_group) if not self.flat_baseline else -1
        self.poststep_observed_bottleneck_history.append(post_obs_bottleneck)
        self.step += 1
        return true_j

    def run(self, policy_func):
        for _ in range(self.n_steps):
            obs = self.observe()
            # Decision-time observed signals
            decision_obs_j = self.compute_J(obs)
            self.decision_obs_J_history.append(decision_obs_j)
            decision_obs_group = np.prod(obs, axis=1)**(1/5) if self.use_geo else np.mean(obs, axis=1)
            decision_obs_bottleneck = np.argmin(decision_obs_group) if not self.flat_baseline else -1
            self.decision_obs_bottleneck_history.append(decision_obs_bottleneck)
            # Policy
            if self.flat_baseline:
                # Greedy flat-sum baseline: all budget to single max safe net eff sub
                flat_effs = np.tile(self.effs, 6)
                net_effs = flat_effs.copy()
                # Poison pill: G1m1 (index 0) net = eff[0] - coupling * eff[0] (for linked G2m1 index 5)
                net_effs[0] -= self.coupling * flat_effs[5]  # Net for boosting G1m1
                best_sub = np.argmax(net_effs)
                if net_effs[best_sub] <= 0: best_sub = np.argmax(flat_effs[1:]) + 1  # Avoid negative
                energy_alloc = np.zeros((6,5))
                g, s = divmod(best_sub, 5)
                energy_alloc[g, s] = self.budget
            else:
                energy_alloc = policy_func(obs, self.effs, self.budget, self.use_geo)
            # Energy accounting
            cheap_mask = self.effs >= 0.333
            spent_cheap = np.sum(energy_alloc[:, cheap_mask])
            self.energy_cheap += spent_cheap
            self.energy_exp += self.budget - spent_cheap
            self.step_agent(energy_alloc)
        total_energy = self.energy_cheap + self.energy_exp
        cheap_pct = (self.energy_cheap / total_energy * 100) if total_energy > 0 else 0
        return {
            'true_J_history': np.array(self.true_J_history),
            'decision_obs_J_history': np.array(self.decision_obs_J_history),
            'poststep_observed_J_history': np.array(self.poststep_observed_J_history),
            'preclamp_min_sub_history': np.array(self.preclamp_min_sub_history),
            'state_min_sub_history': np.array(self.state_min_sub_history),
            'floor_breach_attempts': self.floor_breach_attempts,
            'cheap_energy_pct': cheap_pct,
            'true_bottleneck_history': np.array(self.true_bottleneck_history),
            'decision_obs_bottleneck_history': np.array(self.decision_obs_bottleneck_history),
            'poststep_observed_bottleneck_history': np.array(self.poststep_observed_bottleneck_history)
        }

def jam_policy(obs, effs, budget, use_geo):
    group_scores = np.prod(obs, axis=1)**(1/5) if use_geo else np.mean(obs, axis=1)
    bottleneck = np.argmin(group_scores)
    group_obs = obs[bottleneck]
    marginal = effs / group_obs  # Favors low-value + cheap
    weights = marginal / marginal.sum()
    alloc = np.zeros((6,5))
    alloc[bottleneck] = weights * budget
    return alloc

# Example multi-seed runner (reproduces single run; extend for 50 seeds)
def run_simulation(seed=42):
    np.random.seed(seed)
    env = JAMEnv()
    results = env.run(jam_policy)
    print(f"Final True J: {results['true_J_history'][-1]:.3f}")
    print(f"Min State Sub Ever: {min(results['state_min_sub_history']):.3f}")
    print(f"Breach Attempts: {results['floor_breach_attempts']}")
    print(f"Cheap Energy %: {results['cheap_energy_pct']:.1f}")

# Run a test
run_simulation()
