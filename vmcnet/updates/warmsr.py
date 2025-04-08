from typing import Tuple, Dict

import jax
import kfac_jax
from kfac_jax import Optimizer as kfac_Optimizer
from ml_collections import ConfigDict
import chex
import jax.numpy as jnp
import optax
import neural_tangents as nt  # type: ignore

import vmcnet.mcmc.position_amplitude_core as pacore
import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.utils.pytree_helpers import (
    tree_reduce_l1,
)

import vmcnet.utils.curvature_tags_and_blocks as curvature_tags_and_blocks

from vmcnet.utils.typing import (
    Array,
    Callable,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    OptimizerState,
    P,
    S,
    PRNGKey,
    PyTree,
    UpdateDataFn,
    ModelApply
)

from .update_param_fns import UpdateParamFn, update_metrics_with_noclip, make_traced_fn_with_single_metrics
from .optax_utils import initialize_optax_optimizer
from typing import NamedTuple

from sklearn.utils.extmath import randomized_svd
import math
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import chex
from typing import Tuple, Optional

class WarmSROptimizerState(NamedTuple):
    opt_state: optax.OptState
    prev_eloc: Array
    prev_O: Array
    prev_U: Array
    prev_X: Array


def construct_svd_update_param_fn(
    energy_data_val_and_grad,
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, centered_local_energies, stats, params_grad = energy_data_val_and_grad(params, position)
        params, optimizer_state = optimizer_apply(
            centered_local_energies,
            params,
            params_grad,
            optimizer_state,
            data,
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn

def initialize_warmsr(
        log_psi_apply: ModelApply[P],
        energy_and_statistics_fn,
        energy_data_val_and_grad,
        params: P,
        get_position_fn: GetPositionFromData[D],
        update_data_fn: UpdateDataFn[D, P],
        learning_rate_schedule: LearningRateSchedule,
        optimizer_config: ConfigDict,
        record_param_l1_norm: bool = False,
        apply_pmap: bool = True,
    ) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
        """Get an update param function and initial state for SVD."""
        warmsr_step = get_svd_step(
            log_psi_apply,
            optimizer_config.damping,
            optimizer_config.mu,
            optimizer_config.srft_rank,
        )

        descent_optimizer = optax.sgd(
            learning_rate=learning_rate_schedule, momentum=0, nesterov=False
        )

        def optimizer_apply(centered_local_energies, params, params_grad, optimizer_state, data):
            # local_energies = local_energies.reshape(-1)
            # centered_local_energies = local_energies - energy

            grad, _, new_prev_eloc, new_prev_O, new_prev_U, new_prev_X = warmsr_step(
                centered_local_energies,
                params,
                params_grad,
                optimizer_state.prev_eloc,
                optimizer_state.prev_O,
                optimizer_state.prev_X,
                get_position_fn(data),
            )

            updates, new_opt_state = descent_optimizer.update(
                grad, optimizer_state.opt_state, params
            )

            if optimizer_config.constrain_norm:
                updates = constrain_norm(
                    updates,
                    optimizer_config.norm_constraint,
                )

            params = optax.apply_updates(params, updates)


            new_state = WarmSROptimizerState(
                opt_state=new_opt_state,
                prev_eloc=new_prev_eloc,
                prev_O=new_prev_O,
                prev_U=new_prev_U,
                prev_X=new_prev_X,
            )

            return params, new_state


        update_param_fn = construct_svd_update_param_fn(
            energy_data_val_and_grad,
            optimizer_apply,
            get_position_fn=get_position_fn,
            update_data_fn=update_data_fn,
            record_param_l1_norm=record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        optax_optimizer_state = initialize_optax_optimizer(
            descent_optimizer, params, apply_pmap=apply_pmap
        )
        optimizer_state = WarmSROptimizerState(opt_state=optax_optimizer_state,
                        prev_eloc=None,
                        prev_O=None,
                        prev_U=None,
                        prev_X=None,
                        )
        return update_param_fn, optimizer_state

def get_svd_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.95,
    srft_rank: int = 500,
):
    """Get the SVD-based natural gradient update function."""
    def flatten_batch_gradients(params_grad, nchains):
        # Get unravel function using the first element of the batch
        flat_example, unravel_fn = ravel_pytree(jax.tree_map(lambda x: x[0], params_grad))

        # Define a function that flattens one sample
        def flatten_one(p):
            flat, _ = ravel_pytree(p)
            return flat

        # Vectorize across chains (i.e. batch dimension)
        flat_grads = jax.vmap(flatten_one)(params_grad)  # shape: (nchains, n_params)

        return flat_grads.T, unravel_fn  # shape: (n_params, nchains)

    def svd_step(
        centered_energies: Array,     # shape: (nchains,)
        params: P,                    # current model parameters
        params_grad: P,               # PyTree of shape [nchains, ...] per leaf
        prev_eloc: Optional[Array],   # previous energy target vector
        prev_O: Optional[Array],      # previous projection matrix (n_params, r)
        prev_X: Optional[Array],      # previous low-rank init matrix
        positions: Array,             # shape: (nchains, ...)
    ) -> Tuple[P, int, Array, Array, Array, Array]:

        nchains = positions.shape[0]

        # Step 1: flatten per-sample grads
        # === fix this
        grads_flat, unravel_fn = flatten_batch_gradients(params_grad, nchains)  # (n_params, nchains)
        # ====

        O = grads_flat
        # Step 2: center O
        O = O - jnp.mean(O, axis=1, keepdims=True)
        O = O - jnp.mean(O, axis=0, keepdims=True)

        # Step 3: center local energy
        epsilon_bar = centered_energies / jnp.sqrt(nchains)  # shape: (nchains,)

        # Step 4: memory-augmented matrix (oa, ea)
        if prev_O is None:
            oa = O
            ea = epsilon_bar
        else:
            oa = jnp.hstack([jnp.sqrt(mu) * prev_O, jnp.sqrt(1 - mu) * O])  # (n_params, r + nchains)
            ea = jnp.concatenate([jnp.sqrt(mu) * prev_eloc, jnp.sqrt(1 - mu) * epsilon_bar], axis=0)

        # Step 5: low-rank SVD
        O_rank = min(srft_rank, oa.shape[1])

        if prev_X is None:
            U, S, V, X = lmsvd(oa, O_rank, maxit=300, X=None)
        else:
            U, S, V, X = ssisvd(oa, O_rank, maxit=10, X=prev_X)

        # Step 6: Truncate with damping
        ratio = S / S[0]
        ind = ratio > damping
        sigma0 = 1.0 / ((damping * jnp.abs(S[0]))**2)

        U_trunc = U[:, ind]
        S_trunc = S[ind]
        V_trunc = V[:, ind]

        # Step 7: Compute parameter-space update
        f = oa @ ea
        uf = U_trunc.T @ f
        uf = uf * ((1.0 / (S_trunc**2)) - sigma0)
        dw_tot = U_trunc @ uf + sigma0 * f  # shape: (n_params,)

        # Step 8: Update memory terms
        prev_O = U_trunc @ jnp.diag(S_trunc)
        prev_eloc = V_trunc.T @ ea

        # Step 9: Return update as PyTree
        update_tree = unravel_fn(dw_tot / jnp.sqrt(nchains))

        return update_tree, len(S_trunc), prev_eloc, prev_O, U, X

    return svd_step

# ==== SVD implementations =====

def ssi(A, X, Y, maxit, k=3):
    def body_fn(i, carry):
        X, Y = carry
        X = A @ Y
        X = jax.lax.cond(
            (i % k == 0) | (i == maxit),
            lambda x: jnp.linalg.qr(x, mode='reduced')[0],
            lambda x: x,
            X,
        )
        Y = A.T @ X
        return X, Y

    X, Y = jax.lax.fori_loop(1, maxit + 1, body_fn, (X, Y))
    return X, Y

def ssisvd(A, r, X=None, maxit=10):
    m, n = A.shape
    l = min(2 * r, r + 10, m, n)

    if X is None:
        key = jax.random.PRNGKey(42)  # replace with a real key if needed
        Y = jax.random.normal(key, (n, l))
        X = A @ Y
    else:
        if X.shape[1] < r:
            extra_dim = r - X.shape[1]
            key = jax.random.PRNGKey(123)  # again, use a real key
            extra = jax.random.normal(key, (X.shape[0], extra_dim))
            X = jnp.hstack([X, extra])
        Y = A.T @ X
        X = A @ Y

    Q, _ = jnp.linalg.qr(X, mode='reduced')
    Y = A.T @ Q
    X, Y = ssi(A, Q, Y, maxit)

    QY, R = jnp.linalg.qr(Y, mode='reduced')
    r_diag = jnp.diag(R)
    sign_R = jnp.sign(r_diag)
    r_diag_abs = jnp.abs(r_diag)
    sorted_indices = jnp.argsort(r_diag_abs)[::-1][:r]

    X = X * sign_R[jnp.newaxis, :]
    U = X[:, sorted_indices]
    S = r_diag_abs[sorted_indices]
    V = QY[:, sorted_indices]

    return U, S, V, X


import numpy as np

def lmsvd(A, r, X=None, tol=1e-8, maxit=10, memo=3):
    m, n = A.shape
    if X is None:
        l = min(2 * r, r + 10, m, n)
        Y = np.random.randn(n, l)
        X = A.dot(Y)
    else:
        if X.shape[1] < r:
            extra = np.random.randn(X.shape[0], r - X.shape[1])
            X = np.hstack([X, extra])
        Y = A.T.dot(X)
        X = A.dot(Y)
    Q, _ = np.linalg.qr(X)
    X = Q.copy()
    Y = A.T.dot(X)
    X, Y = lm_lbo(A, X, Y, r, tol, maxit, memo)
    U, S, V = get_svd(X, Y)
    return U[:, :r].copy(), S[:r].copy(), V[:, :r].copy(), X

def get_svd(X, Y):
    Q, R = np.linalg.qr(Y)
    W, S, Z = np.linalg.svd(R.T, full_matrices=False)
    U = X.dot(W)
    V = Q.dot(Z)
    return U, S, V

def lm_lbo(A, X, Y, r, tol, maxit, memo):
    m, _ = X.shape
    n, _ = Y.shape
    k = Y.shape[1]
    mn = min(m, n)
    Lm = k
    rvr = np.zeros(r, dtype=X.dtype)
    rvr0 = np.zeros(r, dtype=X.dtype)
    chg_rvr = 1.0
    chgv = np.zeros(maxit, dtype=X.dtype)
    kktc = np.zeros(maxit, dtype=X.dtype)
    xtrm = np.zeros(maxit, dtype=X.dtype)
    qtol = np.finfo(X.dtype).eps ** min(mn / (40 * k), 1)
    rtol = 5 * max(np.sqrt(tol * qtol), 5 * np.finfo(X.dtype).eps)
    ptol = 5 * max(tol, np.sqrt(np.finfo(X.dtype).eps))
    if k < r:
        raise ValueError("Working size too small")
    Xm = np.zeros((m, (1 + memo) * k), dtype=X.dtype)
    Ym = np.zeros((n, (1 + memo) * k), dtype=X.dtype)
    Xm[:, k:2*k] = X
    Ym[:, k:2*k] = Y
    for it in range(1, maxit + 1):
        SX = X.copy()
        AY = A.dot(Y)
        Q, _ = np.linalg.qr(AY)
        X = Q.copy()
        Y = A.T.dot(X)
        if Lm == 0 or it <= 3:
            SYTY = SX.T.dot(AY)
            SYTY = 0.5 * (SYTY + SYTY.T)
            tE, tU = np.linalg.eigh(SYTY)
            rvr0 = rvr.copy()
            rvr = tE[-r:].copy()
            chg_rvr = np.linalg.norm(rvr0 - rvr) / np.linalg.norm(rvr)
            AY = AY.dot(tU)
            SX = SX.dot(tU)
        xtrm[it - 1] = Lm / k
        chgv[it - 1] = chg_rvr
        if chg_rvr < rtol:
            kkt = AY[:, -r:] - SX[:, -r:] * rvr[np.newaxis, :]
            kktcheck = np.max(np.sqrt(np.sum(kkt**2, axis=0))) / max(tol, rvr[-1])
            if kktcheck < ptol:
                break
            kktc[it - 1] = kktcheck
        else:
            if it == 1:
                kktc[it - 1] = 0
            else:
                kktc[it - 1] = kktc[it - 2]
        xtrm[it - 1] = Lm / k
        if Lm == 0:
            continue
        Xm[:, :k] = X
        Ym[:, :k] = Y
        Im = np.arange(k, k + Lm)
        T = X.T.dot(Xm[:, Im])
        Xm_sub = Xm[:, Im]
        XT = X.dot(T)
        Px = Xm_sub - XT
        Ym_sub = Ym[:, Im]
        YT = Y.dot(T)
        Py = Ym_sub - YT
        T_mat = Px.T.dot(Px)
        if Lm > 50:
            dT = np.diag(T_mat)
            sdT = np.sort(dT)[::-1]
            idx = np.argsort(dT)[::-1]
            L = int(np.sum(sdT > 5e-8))
            if L < 0.95 * Lm:
                Lm = L
                Icut = idx[:Lm]
                Py = Py[:, Icut]
                T_mat = T_mat[np.ix_(Icut, Icut)]
        ev, U_eig = np.linalg.eigh(T_mat)
        e_tol = min(np.sqrt(np.finfo(T_mat.dtype).eps), tol)
        cut = None
        for i, val in enumerate(ev):
            if val > e_tol:
                cut = i
                break
        if cut is None:
            Lm = 0
            continue
        L_val = Lm - cut + 1
        dv = 1.0 / np.sqrt(ev[cut:])
        T_1 = U_eig[:, cut:] @ np.diag(dv)
        Yo = np.hstack([Y, Py @ T_1])
        T_2 = Yo.T.dot(Yo)
        T_2 = 0.5 * (T_2 + T_2.T)
        D, U2 = np.linalg.eigh(T_2)
        Y = Yo @ U2[:, -k:]
        Lm = max(0, int(round(L_val / k))) * k
        if it < memo:
            Lm += k
        if Lm > 0:
            Xm[:, k:k+Lm] = Xm[:, :Lm]
            Ym[:, k:k+Lm] = Ym[:, :Lm]
        rvr0 = rvr.copy()
        rvr = D[-r:].copy()
        chg_rvr = np.linalg.norm(rvr - rvr0) / np.linalg.norm(rvr)
    return X, Y


from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
