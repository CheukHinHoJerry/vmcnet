from typing import Tuple

import jax
import kfac_jax
from kfac_jax import Optimizer as kfac_Optimizer
from ml_collections import ConfigDict
import chex
import jax.numpy as jnp

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
    PRNGKey,
    PyTree,
    UpdateDataFn,
)

from .update_param_fns import UpdateParamFn, update_metrics_with_noclip

def construct_svd_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)

        energy, local_energies, stats = energy_and_statistics_fn(params, position)

        params, optimizer_state = optimizer_apply(
            energy,
            local_energies,
            params,
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

def initialize_svd(
        log_psi_apply: ModelApply[P],
        energy_and_statistics_fn,
        params: P,
        get_position_fn: GetPositionFromData[D],
        update_data_fn: UpdateDataFn[D, P],
        learning_rate_schedule: LearningRateSchedule,
        optimizer_config: ConfigDict,
        record_param_l1_norm: bool = False,
        apply_pmap: bool = True,
    ) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
        """Get an update param function and initial state for SVD."""
        spring_step = get_svd_step(
            log_psi_apply,
            optimizer_config.damping,
            optimizer_config.mu,
        )

        descent_optimizer = optax.sgd(
            learning_rate=learning_rate_schedule, momentum=0, nesterov=False
        )

        def optimizer_apply(energy, local_energies, params, optimizer_state, data):
            centered_local_energies = local_energies - energy
            grad = svd_step(
                centered_local_energies,
                prev_eloc,
                prev_O,
                prev_update(optimizer_state),
                get_position_fn(data),
            )

            updates, optimizer_state = descent_optimizer.update(
                grad, optimizer_state, params
            )

            if optimizer_config.constrain_norm:
                updates = constrain_norm(
                    updates,
                    optimizer_config.norm_constraint,
                )

            params = optax.apply_updates(params, updates)
            return params, optimizer_state

        update_param_fn = construct_svd_update_param_fn(
            energy_and_statistics_fn,
            optimizer_apply,
            get_position_fn=get_position_fn,
            update_data_fn=update_data_fn,
            record_param_l1_norm=record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        optimizer_state = initialize_optax_optimizer(
            descent_optimizer, params, apply_pmap=apply_pmap
        )

        return update_param_fn, optimizer_state

from sklearn.utils.extmath import randomized_svd
import math
def get_svd_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.95,
    srft_rank: int = 500,
):
    """Get the SVD update function."""
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def svd_step(
        centered_energies: P,
        params: P,
        prev_eloc,
        prev_O,
        prev_X,
        positions: Array,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]

        O = kernel_fn(positions, positions, "ntk", params) / nchains
        O = O - jnp.mean(T, axis=0, keepdims=True)
        O = O - jnp.mean(T, axis=1, keepdims=True)
        epsilon_bar = centered_energies / jnp.sqrt(nchains)

        oa = jnp.hstack([jnp.sqrt(mu) * prev_O, jnp.sqrt(1-mu) * O.T])
        ea = jnp.concatenate([jnp.sqrt(mu) * prev_eloc, jnp.sqrt(1-mu) * epsilon_bar], axis=0)

        O_rank = min(srft_rank, oa.shape[1])
        if i == 1:
            U, S, V = lmsvd(oa, O_rank, maxit=300, X=prev_X)
        else:
            U, S, V = ssisvd(oa, O_rank, maxit=10, X=prev_X)

        ratio = S / S[0]
        ind = ratio > damping
        sigma0 = 1.0 / ((damping * jnp.abs(S[0]))**2)
        U_trunc = U[:, :ind]
        S_trunc = S[:ind]
        V_trunc = V[:, :ind]
        f = oa.T @ ea
        uf = U_trunc.T @ f        # shape: (ind, 1)
        uf = uf * ((1.0 / (S_trunc**2)).reshape(-1, 1) - sigma0)
        dw_tot = U_trunc @ uf + sigma0 * f  # shape: (nchains, 1)
        prev_O = U_trunc @ jnp.diag(S_trunc)
        prev_eloc = V_trunc.T @ ea
        return dw_tot, len(ind), prev_eloc, prev_O, U

    return svd_step


import numpy as np

def SSI(A, X, Y, maxit, k=3):
    for iter in range(1, maxit + 1):
        X = np.dot(A, Y)
        if iter % k == 0 or iter == maxit:
            Q, _ = np.linalg.qr(X, mode='reduced')
            X = Q
        Y = np.dot(A.T, X)
    return X, Y

def ssisvd(A, r, X=None, maxit=10):
    m, n = A.shape
    if X is None:
        l = min(2 * r, r + 10, m, n)
        Y = np.random.randn(n, l)
        X = np.dot(A, Y)
    else:
        if X.shape[1] < r:
            extra = np.random.randn(X.shape[0], r - X.shape[1])
            X = np.hstack([X, extra])
        Y = np.dot(A.T, X)
        X = np.dot(A, Y)
    Q, _ = np.linalg.qr(X, mode='reduced')
    Y = np.dot(A.T, Q)

    X, Y = SSI(A, X, Y, maxit)

    QY, R = np.linalg.qr(Y, mode='reduced')
    r_diag = np.diag(R)
    sign_R = np.sign(r_diag)
    r_diag = np.abs(r_diag)
    sorted_indices = np.argsort(r_diag)[::-1][:r]

    X = X * sign_R[np.newaxis, :]

    U = X[:, sorted_indices].copy()
    S = r_diag[sorted_indices].copy()
    V = QY[:, sorted_indices].copy()

    return U, S, V


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
    return U[:, :r].copy(), S[:r].copy(), V[:, :r].copy()

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
