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

class RSSROptimizerState(NamedTuple):
    opt_state: optax.OptState
    prev_eloc: Array
    prev_O: Array


def construct_sketch_update_param_fn(
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

def initialize_rssr(
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
    """Get an update param function and initial state for SKETCH."""
    rssr_step = get_sketch_step(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
        optimizer_config.srft_rank,
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )


    def optimizer_apply(centered_local_energies, params, params_grad, optimizer_state, data):

        grad, _, new_prev_eloc, new_prev_O = rssr_step(
            centered_local_energies,
            params,
            params_grad,
            optimizer_state.prev_eloc,
            optimizer_state.prev_O,
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


        new_state = RSSROptimizerState(
            opt_state=new_opt_state,
            prev_eloc=new_prev_eloc,
            prev_O=new_prev_O,
        )

        return params, new_state

    update_param_fn = construct_sketch_update_param_fn(
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
    optimizer_state = RSSROptimizerState(opt_state=optax_optimizer_state,
                    prev_eloc=None,
                    prev_O=None,
                    )
    return update_param_fn, optimizer_state

from sklearn.utils.extmath import randomized_svd
import math
import numpy as np

def get_sketch_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.95,
    srft_rank: int = 500,
):
    """Get the SKETCH update function."""
    
    def flatten_batch_gradients(params_grad, nchains):
        flat_example, unravel_fn = ravel_pytree(jax.tree_map(lambda x: x[0], params_grad))
        flat_grads = []
        for i in range(nchains):
            sample_i = jax.tree_map(lambda x: x[i], params_grad)
            flat_i, _ = ravel_pytree(sample_i)
            flat_grads.append(flat_i)
        return jnp.stack(flat_grads).T, unravel_fn  # shape: (n_params, nchains)

    def sketch_step(
        centered_energies: P,
        params: P,
        params_grad: P,
        prev_eloc,
        prev_O,
        positions,
    ) -> Tuple[P, int, np.ndarray, np.ndarray]:
        nchains = positions.shape[0]

        # Step 1: flatten per-sample grads
        grads_flat, unravel_fn = flatten_batch_gradients(params_grad, nchains)  # (n_params, nchains)
        O = grads_flat

        # Step 2: center O
        O = O - jnp.mean(O, axis=1, keepdims=True)
        O = O - jnp.mean(O, axis=0, keepdims=True)

        epsilon_bar = centered_energies / jnp.sqrt(nchains)

        # Step 3: memory-augmented matrix (oa, ea)
        if prev_O is None:
            oa = O
            ea = epsilon_bar
        else:
            oa = jnp.hstack([jnp.sqrt(mu) * prev_O, jnp.sqrt(1 - mu) * O])
            ea = jnp.concatenate([jnp.sqrt(mu) * prev_eloc, jnp.sqrt(1 - mu) * epsilon_bar], axis=0)

        # Convert to NumPy for SVD
        oa, ea = np.array(oa), np.array(ea)

        # Step 4: randomized SVD
        O_rank = min(srft_rank, oa.shape[1])
        U, S, Vt = randomized_svd(oa, O_rank, n_iter=20, random_state=None)

        # Step 5: Truncate
        ratio = S / S[0]
        ind_mask = ratio > damping
        num_retained = np.sum(ind_mask)

        sigma0 = 1.0 / ((damping * abs(S[0]))**2)
        U_trunc = U[:, ind_mask]
        S_trunc = S[ind_mask]
        V_trunc = Vt[ind_mask, :]

        # Step 6: Compute update
        f = oa @ ea
        uf = U_trunc.T @ f
        uf = uf * ((1.0 / (S_trunc**2)) - sigma0)
        dw_tot = U_trunc @ uf + sigma0 * f

        # Step 7: Update memory
        prev_O = U_trunc @ np.diag(S_trunc)
        prev_eloc = V_trunc @ ea

        # Step 8: Unflatten update
        update_tree = unravel_fn(jnp.array(dw_tot / np.sqrt(nchains)))

        return update_tree, int(num_retained), prev_eloc, prev_O

    return sketch_step


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
