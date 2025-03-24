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

def construct_sketch_update_param_fn(
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

def initialize_sketch(
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
    """Get an update param function and initial state for SKETCH."""
    spring_step = get_sketch_step(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        centered_local_energies = local_energies - energy
        grad = sketch_step(
            centered_local_energies,
            params,
            prev_eloc,
            prev_O,
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

    update_param_fn = construct_sketch_update_param_fn(
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
def get_sketch_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.95,
    srft_rank: int = 500,
):
    """Get the SKETCH update function."""
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def sketch_step(
        centered_energies: P,
        params: P,
        prev_eloc,
        prev_O,
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
        U, S, Vt = randomized_svd(oa, O_rank, n_iter=20, random_state=None)
        ratio = S / S[0]
        ind = ratio > damping
        sigma0 = 1.0 / ((damping * jnp.abs(S[0]))**2)
        U_trunc = U[:, :ind]
        S_trunc = S[:ind]
        V_trunc = Vt[:ind, :]
        f = oa.T @ ea
        uf = U_trunc.T @ f        # shape: (ind, 1)
        uf = uf * ((1.0 / (S_trunc**2)).reshape(-1, 1) - sigma0)
        dw_tot = U_trunc @ uf + sigma0 * f  # shape: (nchains, 1)
        prev_O = U_trunc @ jnp.diag(S_trunc)
        prev_eloc = V_trunc @ ea
        return dw_tot, len(ind), prev_eloc, prev_O

    return sketch_step
