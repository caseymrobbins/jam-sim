"""
JAM Optimizer — log(min()) topology-following optimizer
- No clamping, no epsilon, no softmin, no minmax
- Follows the frontier topology rather than pure gradient direction
- Plateau detection with directional feelers to escape local optima
- Constraints live inside min() only
- Expected pattern: smooth rapid progress, no oscillation regardless of noise
"""

import numpy as np


def log_min(metrics: np.ndarray) -> float:
    """
    Pure log(min()) with no safety features baked in.
    Caller is responsible for ensuring min > 0.
    """
    m = np.min(metrics)
    if m <= 0:
        raise ValueError(f"min(metrics) = {m}. All metrics must be > 0.")
    return np.log(m)


def log_min_gradient(metrics: np.ndarray, param_jacobian: np.ndarray) -> np.ndarray:
    """
    Gradient of log(min(metrics)) with respect to parameters.

    Chain rule:
        d/dθ log(min(m)) = (1 / min(m)) * d/dθ min(m)

    d/dθ min(m) is the gradient of the bottleneck metric only.
    All other metrics contribute zero gradient — they are not the constraint.

    param_jacobian: shape (n_metrics, n_params)
        Row i is d(metric_i)/d(params)
    """
    bottleneck_idx = np.argmin(metrics)
    bottleneck_val = metrics[bottleneck_idx]

    # Gradient flows only through the bottleneck
    grad = (1.0 / bottleneck_val) * param_jacobian[bottleneck_idx]
    return grad


class JAMOptimizer:
    """
    Topology-following optimizer for log(min()).

    Phase 1 — Frontier climbing:
        Adam-style adaptive moment estimation but gradient direction
        is projected onto the improvement frontier. When the bottleneck
        switches, the optimizer smoothly pivots to the new constraint
        rather than oscillating.

    Phase 2 — Plateau detection + feeler search:
        When progress drops below plateau_threshold for patience steps,
        send feelers in n_feelers directions, step_size distance each.
        Move to the best feeler position if it improves the objective.
        Return to Phase 1 from new position.
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        plateau_threshold: float = 1e-6,
        plateau_patience: int = 20,
        feeler_n: int = 16,
        feeler_step: float = 0.05,
        feeler_depth: int = 5,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.feeler_n = feeler_n
        self.feeler_step = feeler_step
        self.feeler_depth = feeler_depth

        # Adam state
        self.m = None   # first moment
        self.v = None   # second moment
        self.t = 0      # step counter

        # Floor state — starts at zero, raised proportionally to surplus on plateau
        self.floors = None
        self.floor_raise_rate = 0.1  # fraction of surplus added to floor each plateau

        # Plateau detection state
        self.recent_values = []
        self.plateau_counter = 0

    def _adam_step(self, grad: np.ndarray) -> np.ndarray:
        """Standard Adam update, returns parameter delta."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # No epsilon — division safe because v_hat > 0 when grad != 0
        delta = self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return delta

    def _raise_floors_proportional(self, metrics: np.ndarray):
        """
        On plateau, raise each dimension's floor proportionally to its surplus.
        Surplus = current metric value - current floor.
        Dimensions coasting high get their floors raised more aggressively.
        The bottleneck gets raised least because it has the least surplus.
        This re-introduces pressure across the whole system automatically.
        """
        if self.floors is None:
            self.floors = np.zeros(len(metrics))

        surplus = metrics - self.floors
        # Surplus should always be >= 0 if floors are maintained correctly
        surplus = np.maximum(surplus, 0.0)

        raise_amount = self.floor_raise_rate * surplus
        self.floors = self.floors + raise_amount

        if np.any(raise_amount > 0):
            print(f"  [floors raised] new floors: {self.floors.round(4)}")

    def _is_plateau(self, current_value: float) -> bool:
        """Returns True when progress has stalled."""
        self.recent_values.append(current_value)
        if len(self.recent_values) > self.plateau_patience:
            self.recent_values.pop(0)

        if len(self.recent_values) < self.plateau_patience:
            return False

        improvement = max(self.recent_values) - min(self.recent_values)
        return improvement < self.plateau_threshold

    def _feeler_search(
        self,
        params: np.ndarray,
        metric_fn,
        current_value: float
    ) -> np.ndarray:
        """
        Send feelers in n_feelers evenly spaced directions.
        Each feeler walks feeler_depth steps of feeler_step size.
        Returns the best position found, or current params if none improve.
        """
        n = len(params)
        best_params = params.copy()
        best_value = current_value

        # Generate evenly spaced unit directions
        # For high dimensional spaces use random orthogonal directions
        if n <= self.feeler_n:
            # Use identity basis + random complement
            directions = [np.eye(n)[i] for i in range(n)]
            while len(directions) < self.feeler_n:
                d = np.random.randn(n)
                d /= np.linalg.norm(d)
                directions.append(d)
        else:
            directions = []
            for _ in range(self.feeler_n):
                d = np.random.randn(n)
                d /= np.linalg.norm(d)
                directions.append(d)

        for direction in directions:
            probe = params.copy()
            for step in range(1, self.feeler_depth + 1):
                probe = probe + direction * self.feeler_step

                metrics = metric_fn(probe)
                if np.min(metrics) <= 0:
                    break  # Hit invalid region, stop this feeler

                value = log_min(metrics)
                if value > best_value:
                    best_value = value
                    best_params = probe.copy()

        return best_params

    def step(
        self,
        params: np.ndarray,
        metrics: np.ndarray,
        param_jacobian: np.ndarray,
        metric_fn=None,
    ) -> np.ndarray:
        """
        Single optimizer step.

        params:          current parameter vector, shape (n_params,)
        metrics:         current metric values, shape (n_metrics,)
                         ALL must be > floors (or > 0 if floors not yet set)
        param_jacobian:  d(metrics)/d(params), shape (n_metrics, n_params)
        metric_fn:       callable(params) -> metrics array
                         required for plateau feeler search

        Returns updated params.
        """
        # Initialize Adam state and floors on first call
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        if self.floors is None:
            self.floors = np.zeros(len(metrics))

        # Measure against floors — the optimizer sees headroom not raw values
        headroom = metrics - self.floors
        if np.min(headroom) <= 0:
            raise ValueError(f"A metric has fallen to or below its floor. headroom: {headroom}")

        current_value = log_min(headroom)  # optimize headroom not raw metrics

        # Phase 2: plateau detection
        if self._is_plateau(current_value):
            self.plateau_counter += 1

            # Raise floors proportionally to surplus before feeler search
            # Dimensions with more headroom above their floor get raised more
            # This re-introduces pressure without manually intervening
            self._raise_floors_proportional(metrics)

            if metric_fn is not None:
                new_params = self._feeler_search(params, metric_fn, current_value)
                if not np.array_equal(new_params, params):
                    # Reset Adam momentum from new position
                    self.m = np.zeros_like(params)
                    self.v = np.zeros_like(params)
                    self.recent_values = []
                    return new_params

        # Phase 1: frontier climbing via Adam
        grad = log_min_gradient(headroom, param_jacobian)
        delta = self._adam_step(grad)
        new_params = params + delta

        return new_params


# ─────────────────────────────────────────────
# Example usage / smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Toy problem: 4 metrics, each is a linear function of 4 parameters.
    True optimum is when all metrics are balanced.
    We expect smooth convergence with no oscillation.
    """
    np.random.seed(42)

    n_params = 4
    n_metrics = 4

    # Random linear metric functions: metrics = W @ params + b
    W = np.abs(np.random.randn(n_metrics, n_params)) * 0.5 + 0.1
    b = np.ones(n_metrics) * 0.1

    def metric_fn(p):
        return W @ p + b

    def jacobian_fn(p):
        return W  # constant for linear case

    # Start with imbalanced params
    params = np.array([2.0, 0.5, 1.5, 0.3])

    optimizer = JAMOptimizer(
        lr=0.05,
        plateau_patience=15,
        feeler_n=12,
        feeler_step=0.03,
        feeler_depth=8
    )

    print(f"{'Step':>6}  {'log(min)':>10}  {'min_headroom':>14}  {'bottleneck':>12}  floors")
    print("-" * 75)

    for step in range(200):
        metrics = metric_fn(params)
        headroom = metrics - (optimizer.floors if optimizer.floors is not None else np.zeros(n_metrics))

        if np.min(headroom) <= 0:
            print(f"Step {step}: headroom hit floor boundary, stopping.")
            break

        jac = jacobian_fn(params)

        if step % 20 == 0:
            floors = optimizer.floors if optimizer.floors is not None else np.zeros(n_metrics)
            bottleneck = np.argmin(headroom)
            value = log_min(headroom)
            print(f"{step:>6}  {value:>10.4f}  {np.min(headroom):>14.4f}  metric_{bottleneck:>2}      {floors.round(2)}")

        params = optimizer.step(params, metrics, jac, metric_fn=metric_fn)

    final_metrics = metric_fn(params)
    final_floors = optimizer.floors
    final_headroom = final_metrics - final_floors
    print(f"\nFinal metrics:  {final_metrics.round(4)}")
    print(f"Final floors:   {final_floors.round(4)}")
    print(f"Final headroom: {final_headroom.round(4)}")
    print(f"Final log(min(headroom)): {log_min(final_headroom):.4f}")
    print(f"Plateau escapes attempted: {optimizer.plateau_counter}")
