import dataclasses
import functools
import hashlib
import json
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.action_freeze as _action_freeze
import openpi.training.config as _config
import openpi.training.data_cursor as _data_cursor
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
from openpi.training.paper_metrics import PaperMetricsWriter
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def needs_fixed_training_reference(config):
    return (getattr(config.model, "rapr_train_action_expert", False)
            and getattr(config.model, "rapr_nonregression_weight", 0.0) > 0)


def init_fixed_reference(config, mesh):
    """Load the original reference only when its training penalty is enabled."""
    if not needs_fixed_training_reference(config):
        if getattr(config.model, "rapr_train_action_expert", False):
            logging.info("Independent training reference and non-regression loss disabled; no reference loaded")
        return None, None
    if config.rapr_reference_params is None:
        raise ValueError("Joint Con1/action training requires the fixed original checkpoint")
    reference_config = dataclasses.replace(
        config.model, use_rapr=False, rapr_paper_orthogonal=False,
        rapr_train_action_expert=False, rapr_learnable_alpha=False, rapr_late_layer_count=0)
    abstract_model = nnx.eval_shape(reference_config.create, jax.random.key(0))
    graphdef, state = nnx.split(abstract_model)
    state = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, jnp.bfloat16), state)
    loader = _weight_loaders.CheckpointWeightLoader(
        config.rapr_reference_params, require_complete=True, missing_regex="$^")
    loaded = _load_weights_and_validate(loader, state.to_pure_dict())
    if any("rapr_" in path for path in traverse_util.flatten_dict(loaded, sep="/")):
        raise ValueError("Original reference unexpectedly contains Con1 parameters")
    state.replace_by_pure_dict(loaded)
    reference = training_utils.FrozenReference(params=state, model_def=graphdef)
    reference_sharding = sharding.fsdp_sharding(reference, mesh)
    reference = jax.device_put(reference, reference_sharding)
    logging.info("Loaded immutable original reference from %s", config.rapr_reference_params)
    return reference, reference_sharding


def scale_paper_updates(config, updates):
    """Scale AdamW updates, not raw gradients (which Adam would normalize)."""
    if not getattr(config.model, "rapr_paper_orthogonal", False):
        return updates
    if getattr(config.model, "rapr_train_action_expert", False):
        pattern = nnx_utils.PathRegex(
            ".*(llm.*_1|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|"
            "state_proj|action_time_mlp_in|action_time_mlp_out).*"
        )
        updates = nnx_utils.state_map(
            updates, pattern, lambda p: p.replace(p.value * config.rapr_action_lr_scale))
    return nnx_utils.state_map(
        updates, nnx_utils.PathRegex(".*rapr_router/alpha_logit.*"),
        lambda p: p.replace(p.value * config.rapr_alpha_lr_scale))


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    reference: training_utils.FrozenReference | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    minimal_diagnostics = (config.rapr_minimal_diagnostics
                           and getattr(config.model, "rapr_paper_orthogonal", False))

    def mask_early_action_layers(tree):
        """Zero slices 0:start in scanned Action arrays, preserving frozen blocks."""
        start_layer = config.model.change_joint_start_layer

        def mask(variable):
            value = variable.value
            if value.ndim < 1 or value.shape[0] <= start_layer:
                raise ValueError(f"Unexpected scanned Action value shape: {value.shape}")
            return variable.replace(value.at[:start_layer].set(0))

        return nnx_utils.state_map(
            tree,
            nnx_utils.PathRegex(".*llm/layers/.*_1.*"),
            mask,
        )

    def scale_late_action_updates(tree):
        """Apply the configured LR ratio only to active late Action slices."""
        start_layer = config.model.change_joint_start_layer
        scale = config.action_change_late_action_lr_scale

        def transform(variable):
            value = variable.value
            if value.ndim < 1 or value.shape[0] <= start_layer:
                raise ValueError(f"Unexpected scanned Action value shape: {value.shape}")
            value = value.at[:start_layer].set(0)
            value = value.at[start_layer:].multiply(scale)
            return variable.replace(value)

        return nnx_utils.state_map(
            tree,
            nnx_utils.PathRegex(".*llm/layers/.*_1.*"),
            transform,
        )

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    def vjepa_loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if config.rapr_action_train_last_n:
            # This runs INSIDE differentiation. Values/graph are unchanged;
            # early Action weights and non-block projections have no gradient.
            model = nnx.merge(nnx.graphdef(model), _action_freeze.transform_frozen_action(
                config, nnx.state(model), stop_gradient=True))
        if getattr(config.model, "use_point_flow", False) or getattr(config.model, "use_rapr", False):
            flow_loss, vjepa_loss, point_loss, point_metrics = model.compute_all_loss_components(
                rng, observation, actions, train=True, reference_velocity=reference_velocity,
                **({"full_diagnostics": False} if minimal_diagnostics else {}),
            )
            if getattr(config.model, "use_point_flow", False):
                assert point_loss is not None
        else:
            flow_loss, vjepa_loss = model.compute_loss_components(rng, observation, actions, train=True)
            point_loss = None
            point_metrics = {}
        assert vjepa_loss is not None
        if config.model.vjepa_aux_warmup_steps > 0:
            warmup = jnp.minimum(
                state.step.astype(jnp.float32) / config.model.vjepa_aux_warmup_steps,
                1.0,
            )
        else:
            warmup = jnp.asarray(1.0, dtype=jnp.float32)
        vjepa_weight = warmup * config.model.vjepa_aux_weight
        flow_loss = jnp.mean(flow_loss)
        vjepa_loss = jnp.mean(vjepa_loss)
        weighted_vjepa_loss = vjepa_weight * vjepa_loss
        action_weight = getattr(config.model, "point_flow_action_loss_weight", 1.0)
        total_loss = action_weight * flow_loss + weighted_vjepa_loss
        loss_info = {
            "flow_loss": flow_loss,
            "action_loss_weight": action_weight,
            "vjepa_loss": vjepa_loss,
            "vjepa_weight": vjepa_weight,
            "weighted_vjepa_loss": weighted_vjepa_loss,
        }
        if not minimal_diagnostics:
            loss_info["vjepa_cosine"] = 1.0 - vjepa_loss
        if getattr(config.model, "use_rapr", False):
            rapr_loss = jnp.mean(point_metrics["rapr_prediction_loss"])
            weighted_rapr_loss = config.model.rapr_loss_weight * rapr_loss
            total_loss = total_loss + weighted_rapr_loss
            if config.model.rapr_prediction_reduction == "mean":
                # Cheap loss bookkeeping, not a fourth expensive Con1 diagnostic.
                # The frozen JEPA auxiliary scalar is excluded from this display.
                loss_info["action_prediction_loss"] = action_weight * flow_loss + weighted_rapr_loss
            if config.model.rapr_nonregression_weight > 0:
                nonregression_loss = jnp.mean(point_metrics["rapr_nonregression_loss"])
                weighted_nonregression_loss = config.model.rapr_nonregression_weight * nonregression_loss
                total_loss = total_loss + weighted_nonregression_loss
                loss_info["rapr_nonregression_loss"] = nonregression_loss
                loss_info["weighted_rapr_nonregression_loss"] = weighted_nonregression_loss
            if getattr(config.model, "rapr_paper_orthogonal", False):
                loss_info_extra = {key: jnp.mean(value) for key, value in point_metrics.items()}
                for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
                    fraction = loss_info_extra.get(f"{suite}_sample_fraction")
                    if fraction is not None:
                        loss_info_extra[f"{suite}_flow_loss"] = (
                            loss_info_extra[f"{suite}_flow_numerator"] / jnp.maximum(fraction, 1e-8))
                if config.model.rapr_nonregression_weight == 0:
                    loss_info_extra.pop("rapr_nonregression_loss", None)
                loss_info.update(loss_info_extra)
            loss_info.update(
                {
                    "rapr_prediction_loss": rapr_loss,
                    "weighted_rapr_prediction_loss": weighted_rapr_loss,
                    "rapr_nonregression_weight": jnp.asarray(config.model.rapr_nonregression_weight),
                    "rapr_route_gate": jnp.mean(point_metrics["rapr_route_gate"]),
                }
            )
            if "rapr_weight_min" in point_metrics:
                loss_info["rapr_weight_min"] = jnp.mean(point_metrics["rapr_weight_min"])
        if point_loss is not None:
            point_loss = jnp.mean(point_loss)
            weighted_point_loss = config.model.point_flow_loss_weight * point_loss
            total_loss = total_loss + weighted_point_loss
            loss_info.update(
                {
                    "point_flow_loss": point_loss,
                    "weighted_point_flow_loss": weighted_point_loss,
                    **{f"point_flow_{name}": jnp.mean(value) for name, value in point_metrics.items()},
                }
            )
        return total_loss, loss_info

    def action_change_loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        action_loss, change_loss = model.compute_action_change_loss_components(
            rng, observation, actions, train=True
        )
        action_loss = jnp.mean(action_loss)
        change_loss = jnp.mean(change_loss)
        weighted_change_loss = config.model.change_loss_weight * change_loss
        return action_loss + weighted_change_loss, {
            "flow_loss": action_loss,
            "change_flow_loss": change_loss,
            "change_loss_weight": config.model.change_loss_weight,
            "weighted_change_flow_loss": weighted_change_loss,
        }

    def achieved_change_adapter_loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        action_loss, change_loss, inverse_mode = model.compute_achieved_change_adapter_loss_components(
            rng, observation, actions, train=True
        )
        action_loss = jnp.mean(action_loss)
        change_loss = jnp.mean(change_loss)
        weighted_change_loss = config.model.change_loss_weight * change_loss
        return action_loss + weighted_change_loss, {
            "flow_loss": action_loss,
            "change_flow_loss": change_loss,
            "weighted_change_flow_loss": weighted_change_loss,
            "con2_inverse_mode": inverse_mode,
        }

    train_rng = jax.random.fold_in(rng, state.step + config.training_step_offset)
    observation, actions = batch
    reference_velocity = None
    if needs_fixed_training_reference(config):
        if reference is None:
            raise ValueError("Paper Con1 stage 2 requires a fixed original-checkpoint reference")
        reference_model = nnx.merge(reference.model_def, reference.params)
        reference_model.eval()
        reference_velocity = jax.lax.stop_gradient(reference_model.reference_training_velocity(
            train_rng, observation, actions, train=True))

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    if getattr(config.model, "use_achieved_change_adapter", False):
        (loss, loss_info), grads = nnx.value_and_grad(
            achieved_change_adapter_loss_fn, argnums=diff_state, has_aux=True
        )(model, train_rng, observation, actions)
    elif getattr(config.model, "use_action_change_mmdit", False):
        (loss, loss_info), grads = nnx.value_and_grad(action_change_loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )
    elif getattr(config.model, "use_vjepa_aux", False):
        (loss, loss_info), grads = nnx.value_and_grad(vjepa_loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )
    else:
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
        loss_info = {}

    if (
        getattr(config.model, "use_action_change_mmdit", False)
        and getattr(config.model, "change_train_action_late", False)
        and not getattr(config.model, "use_achieved_change_adapter", False)
    ):
        # Gemma stores the 18 Transformer blocks in scanned arrays. Preserve
        # differentiability through the frozen early Action computation so
        # action loss can train Change, but prevent optimizer updates to the
        # first 12 Action parameter slices.
        grads = mask_early_action_layers(grads)

    grads = _action_freeze.transform_frozen_action(config, grads)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    updates = scale_paper_updates(config, updates)
    updates = _action_freeze.transform_frozen_action(config, updates)
    if (
        getattr(config.model, "use_action_change_mmdit", False)
        and getattr(config.model, "change_train_action_late", False)
        and not getattr(config.model, "use_achieved_change_adapter", False)
    ):
        # AdamW adds decoupled weight decay after gradient masking.  Mask the
        # final update too, otherwise the nominally frozen first 12 blocks
        # would still drift at every optimizer step.
        updates = scale_late_action_updates(updates)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        **loss_info,
    }
    if not minimal_diagnostics:
        # Frozen billion-parameter reductions are not needed on every update
        # in the three-diagnostic mode. The CPU checkpoint audit retains them.
        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        info["param_norm"] = optax.global_norm(kernel_params)
    if getattr(config.model, "rapr_paper_orthogonal", False):
        info["con1_learning_rate"] = config.lr_schedule.create()(state.step)
        info["action_learning_rate"] = info["con1_learning_rate"] * (
            config.rapr_action_lr_scale if config.model.rapr_train_action_expert else 0)
        info["alpha_learning_rate"] = info["con1_learning_rate"] * (
            config.rapr_alpha_lr_scale if config.model.rapr_learnable_alpha else 0)
        metric_groups = () if minimal_diagnostics else (
            ("head", ".*rapr_delta_head.*"), ("router", ".*rapr_router.*"),
            ("alpha", ".*alpha_logit.*"),
            ("action", ".*(llm.*_1|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|state_proj).*"),
        )
        for name, pattern in metric_groups:
            selector = nnx_utils.PathRegex(pattern)
            info[f"rapr_{name}_grad_norm"] = optax.global_norm(grads.filter(selector))
            info[f"rapr_{name}_update_norm"] = optax.global_norm(updates.filter(selector))
            info[f"rapr_{name}_relative_update"] = info[f"rapr_{name}_update_norm"] / jnp.maximum(
                optax.global_norm(params.filter(selector)), 1e-30)
    if (
        getattr(config.model, "use_action_change_mmdit", False)
        and getattr(config.model, "change_train_action_late", False)
        and not getattr(config.model, "use_achieved_change_adapter", False)
    ):
        info["late_action_lr_scale"] = jnp.asarray(
            config.action_change_late_action_lr_scale, dtype=jnp.float32
        )
    return new_state, info


def main(config: _config.TrainConfig, *, diagnose_first_step: bool = False, replay_data_cursor: bool = False):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        keep_steps=config.keep_steps,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    if getattr(config.model, "rapr_paper_orthogonal", False):
        data_loader.set_start_batch(config.training_step_offset)
        data_config = data_loader.data_config()
        norm_json = json.dumps({key: dataclasses.asdict(value) for key, value in data_config.norm_stats.items()},
                               default=lambda value: np.asarray(value).tolist(), sort_keys=True)
        order_contract = {
            "sampler": "seekable_numpy_seedsequence_epoch_v1", "seed": config.seed,
            "batch_size": config.batch_size, "training_step_offset": config.training_step_offset,
            "task_index_min": data_config.task_index_min, "task_index_max": data_config.task_index_max,
            "episode_split": data_config.episode_split, "split_seed": data_config.split_seed,
            "validation_fraction": data_config.validation_fraction,
            "normalization_sha256": hashlib.sha256(norm_json.encode()).hexdigest(),
            "teacher_manifest_sha256": hashlib.sha256(
                (epath.Path(data_config.transition_state_root) / "manifest.json").read_bytes()).hexdigest(),
        }
        contract_path = config.checkpoint_dir / "data_order.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != order_contract:
            raise ValueError("Refusing to resume with changed sample ordering or normalization")
        contract_path.write_text(json.dumps(order_contract, indent=2) + "\n")
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        logging.info("Restored full training state at completed update %d", int(train_state.step))
        optimizer_counts = [int(value) for value in jax.tree.leaves(train_state.opt_state)
                            if getattr(value, "shape", None) == ()
                            and jnp.issubdtype(value.dtype, jnp.integer)]
        logging.info("Restored optimizer scalar counters: %s", optimizer_counts)
        if getattr(config.model, "rapr_paper_orthogonal", False):
            data_iter.close()
            data_loader.set_start_batch(config.training_step_offset + int(train_state.step))
            data_iter = iter(data_loader)
            batch = next(data_iter)
            logging.info("Restored seekable data cursor at global batch %d",
                         config.training_step_offset + int(train_state.step))
        elif replay_data_cursor:
            if data_loader.data_config().rlds_data_dir is not None:
                raise ValueError("Data cursor replay is supported only for the seeded Torch loader")
            # The loop already fetched batch zero. Checkpoints contain the
            # number of completed updates, not a serialized data iterator.
            batch = _data_cursor.replay_batches(data_iter, batch, int(train_state.step))
        elif int(train_state.step) > 0:
            logging.warning("Data cursor was not restored; this resume restarts the loader at batch zero")

    reference, reference_sharding = init_fixed_reference(config, mesh)
    if config.rapr_action_train_last_n:
        _action_freeze.validate_start(config, resuming=resuming, completed_updates=int(train_state.step))
        scope = _action_freeze.scope_summary(config, train_state.params)
        logging.info("Effective Action freeze scope: %s", json.dumps(scope))
        (config.checkpoint_dir / "action_freeze_scope.json").write_text(json.dumps(scope, indent=2) + "\n")

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, reference_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    if diagnose_first_step:
        # Separate preparation from device execution without changing train_step.
        # The default training path remains ordinary lazy JIT.
        with sharding.set_mesh(mesh):
            started = time.monotonic()
            logging.info("Runtime diagnostic: trace start")
            traced_step = ptrain_step.trace(train_rng, train_state, batch, reference)
            logging.info("Runtime diagnostic: trace complete in %.2fs", time.monotonic() - started)
            started = time.monotonic()
            logging.info("Runtime diagnostic: lower start")
            lowered_step = traced_step.lower()
            logging.info("Runtime diagnostic: lower complete in %.2fs", time.monotonic() - started)
            started = time.monotonic()
            logging.info("Runtime diagnostic: compile start")
            ptrain_step = lowered_step.compile()
            logging.info("Runtime diagnostic: compile complete in %.2fs", time.monotonic() - started)

    start_step = int(train_state.step)
    metrics_writer = (PaperMetricsWriter(config.checkpoint_dir, completed_updates=start_step)
                      if getattr(config.model, "rapr_paper_orthogonal", False) else None)
    if metrics_writer is not None:
        source_root = epath.Path(__file__).resolve().parent.parent
        sources = ("scripts/train.py", "ops/train_paper_con1.py", "src/openpi/models/pi0.py",
                   "src/openpi/models/pi0_config.py", "src/openpi/training/action_freeze.py",
                   "src/openpi/models/orthogonal_con1.py", "src/openpi/training/config.py",
                   "src/openpi/training/paper_metrics.py", "src/openpi/training/checkpoints.py",
                   "src/openpi/training/orthogonal_targets.py", "src/openpi/training/data_loader.py")
        provenance = {"start_completed_updates": start_step, "unix_time": time.time(),
                      "config": dataclasses.asdict(config),
                      "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                        for name in sources}}
        destination = config.checkpoint_dir / f"segment_{start_step}_{time.time_ns()}.json"
        destination.write_text(json.dumps(provenance, default=repr, indent=2) + "\n")
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        if diagnose_first_step and step == start_step:
            started = time.monotonic()
            logging.info("Runtime diagnostic: first device execution start")
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch, reference)
        if diagnose_first_step and step == start_step:
            jax.block_until_ready((train_state, info))
            logging.info("Runtime diagnostic: first device execution complete in %.2fs", time.monotonic() - started)
        infos.append(info)
        if step % config.log_interval == 0 or (metrics_writer is not None and step == config.num_train_steps - 1):
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if getattr(config.model, "rapr_paper_orthogonal", False):
                for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
                    if f"{suite}_sample_fraction" not in reduced_info:
                        continue
                    fraction = float(reduced_info[f"{suite}_sample_fraction"])
                    reduced_info[f"{suite}_flow_loss"] = (
                        float(reduced_info[f"{suite}_flow_numerator"]) / fraction if fraction > 0 else 0.0)
                if not all(np.isfinite(reduced_info[key]) for key in ("loss", "grad_norm")):
                    raise FloatingPointError(f"Non-finite paper Con1 training state at step {step}: {reduced_info}")
                # Tiny per-channel losses must not be rendered as perpetual 0.0000.
                console_keys = ("loss", "flow_loss", "flow_mse_action7", "rapr_delta_nmse", "rapr_route_gate",
                                "rapr_q_uniform_l1",
                                "rapr_residual_flow7_gain", "rapr_uniform_flow7_gain",
                                "rapr_unconditioned_flow7_gain", "rapr_action_grad_norm", "grad_norm")
                info_str = ", ".join(f"{k}={reduced_info[k]:.6g}" for k in console_keys if k in reduced_info)
                record = {"step": step, "completed_updates": step + 1, "unix_time": time.time(),
                          "global_completed_updates": config.training_step_offset + step + 1,
                          **{key: float(value) for key, value in reduced_info.items()}}
                metrics_writer.append(record)
            else:
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
