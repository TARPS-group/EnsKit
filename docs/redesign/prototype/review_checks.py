"""Regression checks for the review's findings, against the revised prototype."""
import jax, jax.numpy as jnp, numpy as np
import enskit
from enskit import kalman, maps, toy
from enskit.distribution import Ensemble, Gaussian, exact_moment_ensemble, reweight
from enskit.algorithms import eki
from enskit.linalg import PSDDiagonal, Dense

p = toy.linear_gaussian(4, 6)
ens = exact_moment_ensemble(jax.random.key(0), p.prior, 200)
ens = maps.pushforward(ens, p.forward, inputs="u", output="g")

# 1. aligned fast path == general path, same key
j = kalman.gaussian_approximation(ens, {"g": p.noise_cov})
fast = kalman.Matheron().build(ens, j, ("g",))(g=p.y, key=jax.random.key(5))
unaligned = Gaussian({n: j.mean(n) for n in j.names}, {n: j.factor(n) for n in j.names},
                          {"g": p.noise_cov})   # same joint, alignment not declared
gen = kalman.Matheron().build(ens, unaligned, ("g",))(g=p.y, key=jax.random.key(5))
print("fast vs general path max diff:", float(jnp.abs(fast["u"] - gen["u"]).max()))
assert jnp.allclose(fast["u"], gen["u"], atol=1e-10)

# 2. Pathwise with an independent term Q on the target: spread includes Q
Q = PSDDiagonal(jnp.full(4, 4.0))
jq = j.add_noise({"u": Q})
exact_var = jnp.diag(jq.condition({"g": p.y}).cov("u").to_dense())
vs = jnp.stack([jnp.var(kalman.Matheron().build(ens, jq, ("g",))(g=p.y, key=jax.random.key(s))["u"], 0, ddof=1)
                for s in range(20)]).mean(0)
print("target-noise variance: exact", exact_var, "pathwise", vs)
assert jnp.allclose(vs, exact_var, rtol=0.15)

# 3. noise-free on aligned with J = N raises (shape check)
gfull = jax.random.normal(jax.random.key(9), (7, 6))               # full-rank values
small = Ensemble({"u": ens["u"][:7], "g": gfull})              # J=7 > N=6 ok
small.project().condition({"g": p.y})
try:
    Ensemble({"u": ens["u"][:6], "g": gfull[:6]}).project().condition({"g": p.y})
    raise SystemExit("expected a ValueError")
except ValueError as e:
    print("J = N noise-free raises:", str(e)[:70])

# 4. weighted projection with a dominant weight stays finite
g = reweight(ens, jnp.zeros(200).at[0].set(20.0)).project()   # ESS ~ 1 + 1e-6
assert bool(jnp.all(jnp.isfinite(g.factor("u").to_dense())))
try:
    reweight(ens, jnp.zeros(200).at[0].set(60.0)).project()     # ESS 1 to precision
    raise SystemExit("expected a ValueError")
except ValueError as e:
    print("dominant weight raises:", str(e)[:60])

# 5. exact moments jointly across blocks
two = Gaussian.independent({"a": (jnp.zeros(2), PSDDiagonal(jnp.ones(2))),
                                 "b": (jnp.ones(3), PSDDiagonal(jnp.full(3, 2.0)))})
e2 = exact_moment_ensemble(jax.random.key(1), two, 10)
print("cross-block sample cov max:", float(jnp.abs(e2.cov("a", "b").to_dense()).max()))
assert jnp.abs(e2.cov("a", "b").to_dense()).max() < 1e-12

# 6. LocalizedUpdateRule under EKI (noise R / increment is a PSDScaled); unlocated pathwise varies by key
state = eki.EKIState(Ensemble({"u": ens["u"][:20], "theta": ens["u"][:20, :1]}), key=jax.random.key(2))
def fwd(u, theta): return p.forward(u) + theta
loc = kalman.DomainLocalization({"u": jnp.arange(4.0)[:, None], "theta": None},
                                jnp.linspace(0, 3, 6)[:, None], radius=3.0, max_neighbors=4)
s1 = eki.advance(state, fwd, p.y, p.noise_cov, 0.5, update_rule=kalman.LocalizedUpdateRule(kalman.Matheron(), loc))
s2 = eki.advance(state.replace(key=jax.random.key(3)), fwd, p.y, p.noise_cov, 0.5,
                 update_rule=kalman.LocalizedUpdateRule(kalman.Matheron(), loc))
print("LocalizedUpdateRule under EKI ok; unlocated block differs across keys:",
      bool(jnp.any(s1.ensemble["theta"] != s2.ensemble["theta"])))
eki.advance(state, fwd, p.y, p.noise_cov, 0.5, update_rule=kalman.LocalizedUpdateRule(kalman.SymmetricSquareRoot(), loc))

# 7. compress gradient at equal singular values
def f(c):
    gg = Gaussian({"a": jnp.zeros(2)}, {"a": Dense(c * jnp.hstack([jnp.eye(2), jnp.eye(2)]))})
    return jnp.sum(gg.compress().factor("a").to_dense() ** 2)
print("compress grad at [I, I]:", jax.grad(f)(1.0))
assert jnp.isfinite(jax.grad(f)(1.0))
print("all review checks passed")
