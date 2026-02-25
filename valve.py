import torch

class BottleneckActiveSetLS:
    """
    Bottleneck Active-Set optimizer with monotone floor-improving line search.

    Hard active set (no softmin):
      S = { i | V_i <= (1+delta)*Vmin }

    Direction:
      Sum of normalized per-metric gradients over S (optionally critical-weighted),
      then normalized.

    Acceptance:
      Backtracking line search accepts ONLY if min(V) improves (or >= if accept_equal=True).

    NOTE on cost:
      If recompute_per_metric=True, per optimizer step you typically do:
        1 forward for V0 (no_grad) +
        |S| forwards for active-set grads +
        N_ls forwards for line search checks
      This can be expensive when |S| is large.

    NOTE on allow_unused=False:
      We intentionally keep allow_unused=False so a disconnected parameter raises an error.
      If a param has no path to V[i], that's almost always a bug in the closure/model wiring.
    """

    def __init__(
        self,
        params,
        lr=0.2,
        delta=0.02,
        backtrack=0.5,
        max_ls=20,
        weight_mode="equal",          # "equal" or "critical"
        recompute_per_metric=True,    # memory-light but more forwards
        accept_equal=False,           # accept non-decreasing floor
    ):
        self.params = list(params)
        self.lr = float(lr)
        self.delta = float(delta)
        self.backtrack = float(backtrack)
        self.max_ls = int(max_ls)
        self.weight_mode = weight_mode
        self.recompute_per_metric = bool(recompute_per_metric)
        self.accept_equal = bool(accept_equal)

    @torch.no_grad()
    def _get_flat_params(self):
        return torch.cat([p.detach().flatten() for p in self.params])

    @torch.no_grad()
    def _set_flat_params(self, flat):
        idx = 0
        for p in self.params:
            n = p.numel()
            p.copy_(flat[idx:idx + n].view_as(p))
            idx += n

    @staticmethod
    def _flat_from_grads(grads):
        return torch.cat([g.reshape(-1) for g in grads])

    @staticmethod
    def _active_set(V0, delta):
        Vmin0 = torch.min(V0)
        thresh = (1.0 + delta) * Vmin0
        S = (V0 <= thresh).nonzero(as_tuple=False).flatten()
        return S, Vmin0, thresh

    def _weights(self, V0, S, Vmin0, thresh):
        """
        Returns per-index weights for S, and a fallback flag if critical weighting degenerates.
        """
        if self.weight_mode != "critical":
            return torch.ones(S.numel(), dtype=V0.dtype, device=V0.device), False

        denom = float((thresh - Vmin0).item())
        denom_ok = denom > 0.0
        if not denom_ok:
            # All active metrics exactly equal -> critical weighting provides no information.
            return torch.ones(S.numel(), dtype=V0.dtype, device=V0.device), True

        # Hard weighting by closeness to floor (not softmin):
        # w_i = (thresh - V_i) / (thresh - Vmin)
        w = []
        th = float(thresh.item())
        for i in S:
            Vi0 = float(V0[int(i)].item())
            w.append((th - Vi0) / denom)
        return torch.tensor(w, dtype=V0.dtype, device=V0.device), False

    def _grad_unit_for_metric(self, V, i):
        """
        Compute normalized gradient of log(V[i]) w.r.t. params.
        """
        Ji = torch.log(V[int(i)])
        g_list = torch.autograd.grad(
            Ji,
            self.params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,  # intentional: disconnected param should raise
        )
        g = self._flat_from_grads([gi.detach() for gi in g_list])
        gn = torch.linalg.norm(g)
        if gn == 0:
            return None
        return g / gn

    def _direction_from_active_set(self, closure, V0, S, Vmin0, thresh):
        """
        Shared logic to compute direction, regardless of recompute/retain style.
        """
        W, weight_fallback = self._weights(V0, S, Vmin0, thresh)

        grads_unit = []

        if self.recompute_per_metric:
            # Memory-light: recompute V with grad for each active index
            for idx_pos, i in enumerate(S):
                V = closure()  # with grad
                gu = self._grad_unit_for_metric(V, i)
                if gu is None:
                    continue
                grads_unit.append(W[idx_pos] * gu)
        else:
            # Faster: build one graph and reuse it (retain_graph=True externally)
            V = closure()  # with grad, single graph
            for idx_pos, i in enumerate(S):
                Ji = torch.log(V[int(i)])
                g_list = torch.autograd.grad(
                    Ji,
                    self.params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,  # intentional
                )
                g = self._flat_from_grads([gi.detach() for gi in g_list])
                gn = torch.linalg.norm(g)
                if gn == 0:
                    continue
                grads_unit.append(W[idx_pos] * (g / gn))

        if not grads_unit:
            return None, weight_fallback

        d = torch.stack(grads_unit, dim=0).sum(dim=0)
        dn = torch.linalg.norm(d)
        if dn == 0:
            return None, weight_fallback
        return d / dn, weight_fallback

    def step(self, closure):
        """
        closure() must return a 1D tensor V of strictly positive viabilities.
        It must be deterministic for a given parameter state.
        """

        # Count closure calls (so profiling surprises are explicit)
        call_count = {"n": 0}
        def counted_closure():
            call_count["n"] += 1
            return closure()

        # 1) Evaluate V0 without graph
        with torch.no_grad():
            V0 = counted_closure()
            S, Vmin0, thresh = self._active_set(V0, self.delta)

        if S.numel() == 0:
            return {"status": "empty_active_set", "Vmin": float(Vmin0), "closure_calls": call_count["n"]}

        # 2) Compute direction using active-set grads
        with torch.enable_grad():
            d, weight_fallback = self._direction_from_active_set(counted_closure, V0, S, Vmin0, thresh)

        if d is None:
            return {
                "status": "no_grad_or_zero_dir",
                "Vmin": float(Vmin0),
                "active_set_size": int(S.numel()),
                "closure_calls": call_count["n"],
                "weight_fallback": bool(weight_fallback),
            }

        # 3) Monotone line search on raw floor min(V)
        flat0 = self._get_flat_params()
        alpha = self.lr
        accepted = False
        Vmin_new = Vmin0  # explicit even on rejection

        for _ in range(self.max_ls):
            self._set_flat_params(flat0 + alpha * d)

            # No graph needed: acceptance uses values only
            with torch.no_grad():
                V_try = counted_closure()
                Vmin_try = torch.min(V_try)

            ok = (Vmin_try >= Vmin0) if self.accept_equal else (Vmin_try > Vmin0)
            if ok:
                accepted = True
                Vmin_new = Vmin_try
                break

            alpha *= self.backtrack

        if not accepted:
            self._set_flat_params(flat0)

        return {
            "status": "ok" if accepted else "reject",
            "Vmin_before": float(Vmin0),
            "Vmin_after": float(Vmin_new),
            "alpha": float(alpha),
            "active_set_size": int(S.numel()),
            "delta": float(self.delta),
            "weight_mode": self.weight_mode,
            "weight_fallback": bool(weight_fallback),
            "recompute_per_metric": self.recompute_per_metric,
            "closure_calls": int(call_count["n"]),
        }
