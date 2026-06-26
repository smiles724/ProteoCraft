"""
Runtime adapter for the optional ESMFold2 backend.

Heavy Biohub ESMFold2 / transformers objects are loaded lazily and only when
``--model_backend esmfold2`` is selected.  The default Protenix V1/V2 path does
not import this module.
"""

from __future__ import annotations

import logging
import math
import string
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.optim as optim

from bindcraft.hallucination.esmfold2_backend import (
    AA_TO_ESM_IDX,
    ESMFold2ProposalConfig,
    ESMFOLD2_ACTION_GROUP_LOSS_KEYS,
    ESM_AA_ORDER,
    build_esm_verifier_result,
    build_gradient_mask,
    combine_normalized_gradients,
    esm_temperature,
    init_binder_logits,
    logit_delta_mutation_scores,
    propose_sequences_from_logit_delta,
    resolve_lambda_esmc,
)


logger = logging.getLogger(__name__)

CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits
ESMC_MASK_FRACTION = 0.15


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().mean().item()
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _target_sequences(target_seq: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(target_seq, str):
        text = target_seq.strip()
        if "|" in text:
            return [part for part in text.split("|") if part]
        return [text]
    return [str(seq).strip() for seq in target_seq if str(seq).strip()]


def _mean_target_to_binder_pair_iptm(
    pair_chains_iptm: Any,
    *,
    n_target_chains: int,
) -> Optional[float]:
    if pair_chains_iptm is None:
        return None
    tensor = torch.as_tensor(pair_chains_iptm).detach().float()
    if tensor.ndim != 2 or tensor.shape[0] <= n_target_chains or tensor.shape[1] <= n_target_chains:
        return None
    binder_idx = n_target_chains
    values = tensor[:n_target_chains, binder_idx]
    if values.numel() == 0:
        return None
    return _as_float(values)


def _midpoint_bins(device: torch.device) -> torch.Tensor:
    boundaries = torch.linspace(2.0, 52.0, 127, device=device)
    lower = torch.tensor([1.0], device=device)
    upper = torch.tensor([57.0], device=device)
    expanded = torch.cat((lower, boundaries, upper))
    return (expanded[:-1] + expanded[1:]) / 2.0


def _binned_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1.0e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def _masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1], device=y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def _masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


def _contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    *,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    con_loss = _binned_entropy(distogram_logits, bin_distance, cutoff)
    positions = torch.arange(distogram_logits.shape[1], device=distogram_logits.device)
    p_dist = positions[:, None] - positions[None, :]
    if min_sep > 0:
        separation_mask = torch.abs(p_dist) >= min_sep
        binder_mask = torch.logical_and(separation_mask, binder_mask.bool())
    per_residue = _masked_min_k(con_loss, mask=binder_mask, k=num_contacts)
    return _masked_average(per_residue, mask=chain_mask)


def _structure_losses(distogram_logits: torch.Tensor, binder_length: int) -> dict[str, torch.Tensor]:
    bin_distance = _midpoint_bins(distogram_logits.device)
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0

    intra = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )
    inter = _contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=1 - is_binder,
        binder_mask=is_binder,
    )

    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    capped_bins = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(capped_bins), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    glob = F.elu(rg_term - rg_th)

    total = 0.5 * intra + 0.5 * inter + 0.2 * glob
    return {
        "intra_contact_loss": intra,
        "inter_contact_loss": inter,
        "glob_loss": glob,
        "total_loss": total,
    }


def _stageb_structure_loss(
    losses: dict[str, torch.Tensor],
    action_group: Optional[str],
    *,
    group_weight: float = 1.0,
) -> tuple[torch.Tensor, str]:
    base_loss = losses["total_loss"]
    weights = ESMFOLD2_ACTION_GROUP_LOSS_KEYS.get(str(action_group or ""), {})
    selected = []
    proposal_loss = None
    for key, weight in weights.items():
        if key not in losses:
            continue
        term = float(weight) * losses[key]
        proposal_loss = term if proposal_loss is None else proposal_loss + term
        selected.append(key)
    if proposal_loss is None:
        return base_loss, "total_loss"
    return base_loss + float(group_weight) * proposal_loss, "total_loss+" + "+".join(selected)


class ESMFold2RuntimeBackend:
    """ESMFold2 verifier + short proposal-gradient adapter."""

    def __init__(
        self,
        *,
        config: ESMFold2ProposalConfig,
        device: Union[str, torch.device] = "cuda",
        verifier_model: Any = None,
        experimental_model: Any = None,
        esmc_model: Any = None,
        input_builder: Any = None,
        protein_input_cls: Any = None,
        structure_prediction_input_cls: Any = None,
        soft_update_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
        task_type: str = "denovo",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.verifier_model = verifier_model
        self.experimental_model = experimental_model
        self.esmc_model = esmc_model
        self.input_builder = input_builder
        self.protein_input_cls = protein_input_cls
        self.structure_prediction_input_cls = structure_prediction_input_cls
        self.soft_update_fn = soft_update_fn
        self.task_type = task_type
        self.verifier_num_loops = int(config.verifier_num_loops)
        self.verifier_num_sampling_steps = int(config.verifier_num_sampling_steps)

    def verify(
        self,
        *,
        binder_seq: str,
        target_seq: Union[str, Sequence[str]],
        seed: Optional[int] = None,
    ):
        self._ensure_verifier_loaded()
        targets = _target_sequences(target_seq)
        input_spec = self._build_structure_input(targets, binder_seq)
        fold_result = self.input_builder.fold(
            self.verifier_model,
            input_spec,
            num_loops=self.verifier_num_loops,
            num_sampling_steps=self.verifier_num_sampling_steps,
            num_diffusion_samples=1,
            seed=seed,
            complex_id="esmfold2_verify",
        )
        summary = self._summary_from_fold_result(fold_result, n_target_chains=len(targets))
        return build_esm_verifier_result(
            binder_seq=binder_seq,
            target_seq=target_seq,
            summary_confidence=summary,
        )

    def propose(
        self,
        *,
        current_seq: str,
        target_seq: Union[str, Sequence[str]],
        global_step: int,
        K_top: int,
        B_candidates: int,
        T_sample: float,
        max_mutations_per_seq: int,
        seed: Optional[int] = None,
        mutable_mask: Optional[Sequence[bool]] = None,
        action_group: Optional[str] = None,
    ) -> list[str]:
        logits_before, logits_after = self._run_soft_update(
            current_seq=current_seq,
            target_seq=target_seq,
            global_step=global_step,
            seed=seed,
            mutable_mask=mutable_mask,
            action_group=action_group,
        )
        candidates = propose_sequences_from_logit_delta(
            logits_before,
            logits_after,
            current_seq=current_seq,
            mutable_mask=mutable_mask,
            K_top=K_top,
            B_candidates=B_candidates,
            T_sample=T_sample,
            max_mutations_per_seq=max_mutations_per_seq,
        )
        return candidates

    def preview_loss_dict(
        self,
        *,
        binder_seq: str,
        target_seq: Union[str, Sequence[str]],
        seed: Optional[int] = None,
    ) -> dict[str, float]:
        """Compute lightweight distogram losses for Stage-B routing diagnostics.

        This uses the same Experimental proposal trunk as the mutation-gradient
        path, but with the current discrete binder as a one-hot design.  The
        output is merged into the verified node only for routing; it does not
        replace the full ESMFold2 verifier score.
        """

        targets = _target_sequences(target_seq)
        if len(targets) != 1:
            raise ValueError("ESMFold2 preview routing currently supports one target chain")
        self._ensure_proposal_models_loaded()
        design = self._sequence_to_design_one_hot(binder_seq)
        fold_result = self._fold_distogram(
            target_seq=targets[0],
            design=design,
            seed=seed,
        )
        losses = _structure_losses(fold_result["distogram_logits"], len(binder_seq))
        output: dict[str, float] = {}
        for key in ("intra_contact_loss", "inter_contact_loss", "glob_loss"):
            value = losses.get(key)
            scalar = _as_float(value)
            if scalar is not None:
                output[key] = scalar
        return output

    def _ensure_verifier_loaded(self) -> None:
        if self.input_builder is None or self.protein_input_cls is None or self.structure_prediction_input_cls is None:
            from esm.models.esmfold2 import (
                ESMFold2InputBuilder,
                ProteinInput,
                StructurePredictionInput,
            )

            self.input_builder = self.input_builder or ESMFold2InputBuilder()
            self.protein_input_cls = self.protein_input_cls or ProteinInput
            self.structure_prediction_input_cls = (
                self.structure_prediction_input_cls or StructurePredictionInput
            )

        if self.verifier_model is None:
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

            logger.info("Loading ESMFold2 verifier model: %s", self.config.verifier_model_name)
            self.verifier_model = ESMFold2Model.from_pretrained(
                self.config.verifier_model_name,
                esmc_precision="bf16",
            ).to(self.device).eval()

    def _ensure_proposal_models_loaded(self) -> None:
        if self.experimental_model is None:
            from transformers.models.esmfold2.modeling_esmfold2_experimental import (
                ESMFold2ExperimentalModel,
            )

            logger.info(
                "Loading ESMFold2 experimental proposal model: %s",
                self.config.experimental_model_name,
            )
            self.experimental_model = ESMFold2ExperimentalModel.from_pretrained(
                self.config.experimental_model_name,
                load_esmc=False,
            ).to(self.device).eval().requires_grad_(False)
            if hasattr(self.experimental_model, "set_kernel_backend"):
                try:
                    self.experimental_model.set_kernel_backend(None)
                except Exception as exc:  # pragma: no cover - optional kernel path
                    logger.warning("Could not set ESMFold2 kernel backend fallback: %s", exc)

        lambda_esmc = resolve_lambda_esmc(self.config.lambda_esmc, task_type=self.task_type)
        if lambda_esmc != 0.0 and self.esmc_model is None:
            from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM

            model_name = getattr(self.config, "esmc_model_name", "biohub/ESMC-6B")
            logger.info("Loading ESMC sequence-prior model: %s", model_name)
            self.esmc_model = ESMCForMaskedLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            ).to(self.device).eval().requires_grad_(False)

    def _build_structure_input(self, targets: Sequence[str], binder_seq: str) -> Any:
        if len(targets) + 1 > len(CHAIN_IDS):
            raise ValueError("Too many chains for first ESMFold2 backend")
        proteins = []
        for idx, sequence in enumerate(targets):
            proteins.append(self.protein_input_cls(id=CHAIN_IDS[idx], sequence=sequence, msa=None))
        proteins.append(
            self.protein_input_cls(id=CHAIN_IDS[len(targets)], sequence=binder_seq, msa=None)
        )
        return self.structure_prediction_input_cls(sequences=proteins)

    def _summary_from_fold_result(self, fold_result: Any, *, n_target_chains: int) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        plddt = _as_float(getattr(fold_result, "plddt", None))
        ptm = _as_float(getattr(fold_result, "ptm", None))
        iptm = _as_float(getattr(fold_result, "iptm", None))
        pair_chain = _mean_target_to_binder_pair_iptm(
            getattr(fold_result, "pair_chains_iptm", None),
            n_target_chains=n_target_chains,
        )
        if plddt is not None:
            summary["plddt"] = plddt
        if ptm is not None:
            summary["ptm"] = ptm
        if iptm is not None:
            summary["iptm"] = iptm
        if pair_chain is not None:
            summary["pair_chain_iptm"] = pair_chain
        complex_obj = getattr(fold_result, "complex", None)
        if complex_obj is not None and hasattr(complex_obj, "to_mmcif_string"):
            summary["mmcif"] = complex_obj.to_mmcif_string()
        return summary

    def _run_soft_update(
        self,
        *,
        current_seq: str,
        target_seq: Union[str, Sequence[str]],
        global_step: int,
        seed: Optional[int],
        mutable_mask: Optional[Sequence[bool]],
        action_group: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.soft_update_fn is not None:
            return self.soft_update_fn(
                current_seq=current_seq,
                target_seq=target_seq,
                global_step=global_step,
                seed=seed,
                action_group=action_group,
                mutable_mask=mutable_mask,
            )

        targets = _target_sequences(target_seq)
        if len(targets) != 1:
            raise ValueError("ESMFold2 proposal-gradient first version supports one target chain")
        self._ensure_proposal_models_loaded()

        logits = self._init_current_sequence_logits(current_seq, mutable_mask=mutable_mask)
        logits_before = logits.detach().clone()[0]
        gradient_mask = build_gradient_mask(
            current_seq,
            mutable_mask=mutable_mask,
            batch_size=1,
            device=self.device,
        )
        optimizer = optim.SGD([logits], lr=float(self.config.learning_rate))
        lambda_esmc = resolve_lambda_esmc(self.config.lambda_esmc, task_type=self.task_type)

        for local_step in range(int(self.config.soft_steps)):
            optimizer.zero_grad()
            temperature = esm_temperature(
                local_step,
                total_steps=int(self.config.soft_steps),
                temperature_min=float(self.config.temperature_min),
                temperature_max=float(self.config.temperature_max),
            )
            design = F.softmax(logits / temperature, dim=-1)
            fold_result = self._fold_distogram(
                target_seq=targets[0],
                design=design,
                seed=None if seed is None else int(seed) + int(global_step) + local_step,
            )
            losses = _structure_losses(fold_result["distogram_logits"], len(current_seq))
            structure_loss, _ = _stageb_structure_loss(
                losses,
                action_group,
                group_weight=getattr(self.config, "stageb_group_weight", 1.0),
            )
            structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

            esmc_grad = None
            if lambda_esmc != 0.0:
                design_for_esmc = F.softmax(logits / temperature, dim=-1)
                score_mask = gradient_mask.sum(dim=-1) > 0
                esmc_loss = self._compute_esmc_pseudoperplexity(
                    design_for_esmc,
                    score_mask=score_mask,
                )
                esmc_grad = torch.autograd.grad(esmc_loss.mean(), logits)[0]

            logits.grad = combine_normalized_gradients(
                structure_grad,
                esmc_grad,
                gradient_mask=gradient_mask,
                lambda_esmc=lambda_esmc,
            )
            for group in optimizer.param_groups:
                group["lr"] = float(self.config.learning_rate) * temperature
            optimizer.step()

        return logits_before.detach().cpu(), logits.detach().clone()[0].cpu()

    def _init_current_sequence_logits(
        self,
        current_seq: str,
        *,
        mutable_mask: Optional[Sequence[bool]],
    ) -> torch.Tensor:
        logits = init_binder_logits(
            current_seq,
            mutable_mask=mutable_mask,
            batch_size=1,
            random_scale=0.0,
            device=self.device,
        )
        with torch.no_grad():
            from bindcraft.hallucination.esmfold2_backend import AA_TO_ESM_IDX, CYS_INDEX

            for pos, aa in enumerate(current_seq):
                if aa in AA_TO_ESM_IDX and AA_TO_ESM_IDX[aa] != CYS_INDEX:
                    logits[:, pos, AA_TO_ESM_IDX[aa]] = 2.0
        return logits

    def _sequence_to_design_one_hot(self, sequence: str) -> torch.Tensor:
        indices = [AA_TO_ESM_IDX[aa] for aa in sequence]
        return F.one_hot(
            torch.tensor(indices, device=self.device),
            num_classes=len(ESM_AA_ORDER),
        ).unsqueeze(0).float()

    def _fold_distogram(
        self,
        *,
        target_seq: str,
        design: torch.Tensor,
        seed: Optional[int],
    ) -> dict[str, Any]:
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput, prepare_esmfold2_input

        target_one_hot = self._sequence_to_folding_one_hot(target_seq)
        padded_design = F.pad(design, (2, 11), mode="constant", value=0.0)
        token_ids = torch.argmax(padded_design, dim=-1)
        token_to_aa = self._folding_tokens_to_aa()
        designed_seq = "".join(token_to_aa[int(token.item())] for token in token_ids[0])
        input_raw = StructurePredictionInput(
            sequences=[
                ProteinInput(id="A", sequence=target_seq, msa=None),
                ProteinInput(id="B", sequence=designed_seq, msa=None),
            ]
        )
        features, _ = prepare_esmfold2_input(input_raw, seed=seed)
        inputs = {
            key: value.unsqueeze(0).to(self.device)
            for key, value in features.items()
            if isinstance(value, torch.Tensor)
        }
        inputs["res_type_soft"] = torch.cat((target_one_hot, padded_design), dim=1)

        output = self.experimental_model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=int(self.config.num_sampling_steps),
            num_loops=int(self.config.num_loops),
            calculate_confidence=False,
            seed=seed,
        )
        return {"distogram_logits": output["distogram_logits"]}

    @lru_cache(maxsize=1)
    def _folding_tokens(self) -> tuple[str, ...]:
        from esm.models.esmfold2.constants import RES_TYPE_TO_CCD

        return ("<pad>", "-") + tuple(RES_TYPE_TO_CCD[i] for i in range(2, 33))

    @lru_cache(maxsize=1)
    def _folding_tokens_to_aa(self) -> dict[int, str]:
        from esm.models.esmfold2.constants import PROTEIN_3TO1

        return {
            idx: PROTEIN_3TO1[token]
            for idx, token in enumerate(self._folding_tokens())
            if token in PROTEIN_3TO1
        }

    def _sequence_to_folding_one_hot(self, sequence: str) -> torch.Tensor:
        from esm.models.esmfold2.constants import PROTEIN_1TO3

        token_to_idx = {token: idx for idx, token in enumerate(self._folding_tokens())}
        indices = [token_to_idx[PROTEIN_1TO3[aa]] for aa in sequence]
        one_hot = F.one_hot(
            torch.tensor(indices, device=self.device),
            num_classes=len(self._folding_tokens()),
        )
        return one_hot.unsqueeze(0).float()

    def _compute_esmc_pseudoperplexity(
        self,
        binder_design: torch.Tensor,
        *,
        score_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.esmc_model is None:
            raise RuntimeError("ESMC model is required when lambda_esmc is non-zero")
        from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

        device = binder_design.device
        lm_vocab_size = self.esmc_model.config.vocab_size
        model_dtype = self.esmc_model.esmc.embed.weight.dtype
        target_esm = binder_design @ self._folding_to_lm_aa_matrix(device)
        discrete = F.one_hot(torch.argmax(target_esm, dim=-1), num_classes=target_esm.size(-1))
        input_esm = target_esm + (discrete.to(target_esm.dtype) - target_esm).detach()

        input_ids = torch.zeros(
            (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
            dtype=model_dtype,
            device=device,
        )
        tokenizer = ESMCTokenizer()
        input_ids[:, 0, tokenizer.cls_token_id] = 1
        input_ids[:, -1, tokenizer.eos_token_id] = 1
        input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

        if score_mask.ndim == 1:
            score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
        score_mask = score_mask.to(device=device, dtype=torch.bool)
        mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
        mask_token[self.esmc_model.config.mask_token_id] = 1

        losses = []
        for batch_idx in range(binder_design.size(0)):
            position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if position_indices.numel() == 0:
                raise ValueError("ESMC score mask selected zero positions")
            n_passes = int(self.config.esmc_n_passes)
            num_masked = max(1, math.ceil(ESMC_MASK_FRACTION * int(position_indices.numel())))
            random_scores = torch.rand((n_passes, position_indices.numel()), device=device)
            masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
            pass_masks = torch.zeros((n_passes, binder_design.size(1)), dtype=torch.bool, device=device)
            pass_masks[
                torch.arange(n_passes, device=device)[:, None],
                position_indices[masked_offsets],
            ] = True
            masked_sequences = input_ids[batch_idx : batch_idx + 1].repeat(n_passes, 1, 1)
            mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
            masked_sequences[mask_rows, mask_cols + 1] = mask_token
            target_weights = target_esm[batch_idx]
            masked_nlls = []
            for start in range(0, n_passes, int(self.config.esmc_batch_size)):
                stop = min(start + int(self.config.esmc_batch_size), n_passes)
                chunk = masked_sequences[start:stop]
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    hidden, *_ = self.esmc_model.esmc.transformer(
                        chunk @ self.esmc_model.esmc.embed.weight.to(chunk.dtype),
                        sequence_id=None,
                        layers_to_collect=[],
                        output_attentions=False,
                    )
                    logits = self.esmc_model.lm_head(hidden)
                log_probs = logits.log_softmax(dim=-1)[:, 1:-1, 4:24]
                nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(dim=-1)
                masked_nlls.append(nlls[pass_masks[start:stop]])
            losses.append(torch.cat(masked_nlls, dim=0).mean())
        return torch.stack(losses, dim=0)

    def _folding_to_lm_aa_matrix(self, device: torch.device) -> torch.Tensor:
        from esm.models.esmfold2.constants import PROTEIN_1TO3
        from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

        three_to_one = {value: key for key, value in PROTEIN_1TO3.items()}
        folding_aas = [three_to_one[token] for token in self._folding_tokens()[2:22]]
        lm_vocab = sorted(ESMCTokenizer().vocab.items(), key=lambda item: item[1])
        lm_aas = [lm_vocab[i][0] for i in range(4, 24)]
        matrix = torch.zeros(20, 20, device=device)
        for folding_idx, aa in enumerate(folding_aas):
            matrix[folding_idx, lm_aas.index(aa)] = 1.0
        return matrix
