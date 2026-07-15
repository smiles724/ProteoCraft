"""
BindCraft v2 - TCR sequence prior scorer interface.

Provides two optional integration points:
  1. Gradient proposal path: ``soft_prior_loss(sequence_logits, region_mask, metadata)``
     returns a differentiable loss that is added to the proposal loss before
     back-propagation. Gradients therefore flow to the sequence logits (theta).
  2. Verifier / tree-value path: ``score_hard_sequence(sequence, region_mask, metadata)``
     returns a scalar score for a discrete candidate sequence. The score is
     converted to a loss term and added to the verifier total_loss / loss_dict,
     so it is visible in MCTS node metadata and tree rewards.

Real prior backends are selected via ``tcr_prior_model_name`` and plug in
behind the same interface (see TCR_PRIOR_BACKENDS.md):
  - ``tcrlm[:/repo]``     — hliulab/tcrLM masked-LM CDR3 naturalness (hard+soft).
  - ``esmc[:model_id]``   — EvolutionaryScale ESM-C whole-TCR naturalness (hard+soft).
  - ``ligandmpnn[:/repo]``— dauparas/LigandMPNN structure-conditioned CDR prior.
All backends are frozen (never trained/fine-tuned); the prior is a *target*
for the soft loss and a *reward* for the hard verifier, and gradients never
flow into the backend model. If a backend cannot load (missing repo / weights
/ deps / structure), it degrades to a no-op so the main loop is unaffected.

Any other ``tcr_prior_model_name`` (incl. "", "none", "dummy", "null", or a
plain HuggingFace id) uses the original placeholder / HF adapter, which:
  - Loads a user-specified HF model if available.
  - Falls back to a uniform-composition placeholder otherwise.

All integration points are disabled by default; original behavior is preserved
when the corresponding flags are off.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from bindcraft.utils import AA_1LETTER, NUM_AMINO_ACIDS, PROTEIN_RESIDUE_NAMES

logger = logging.getLogger(__name__)

AA_1_TO_IDX = {AA_1LETTER[name]: idx for idx, name in enumerate(PROTEIN_RESIDUE_NAMES)}
IDX_TO_AA1 = [AA_1LETTER[name] for name in PROTEIN_RESIDUE_NAMES]

VALID_TCR_PRIOR_REGIONS = ("all", "cdr", "none")


class TCRPriorStrictError(RuntimeError):
    """Raised in strict mode when a requested real TCR prior backend is not
    fully functional (failed to load, missing real weights, or would fall back
    to dummy/no-op behavior)."""

# Recognized real-backend prefixes for ``tcr_prior_model_name``. Anything else
# (including "", "none", "dummy", "null" or a plain HF id) keeps the original
# ``TCRSequencePriorScorerHF`` behavior for full backward compatibility.
_TCRLM_PREFIXES = ("tcrlm",)
_ESMC_PREFIXES = ("esmc", "esm-c", "esm_c")
_LIGANDMPNN_PREFIXES = ("ligandmpnn", "ligand_mpnn", "ligand-mpnn")


def _region_indices(
    region_mask: Optional[Union[Sequence[bool], torch.Tensor]],
    length: int,
) -> List[int]:
    """Return the list of position indices selected by ``region_mask``."""
    mask = _normalize_region_mask(region_mask, length, torch.device("cpu"))
    return [int(i) for i in torch.nonzero(mask, as_tuple=False).flatten().tolist()]


def _logits_to_sequence(sequence_logits: torch.Tensor) -> str:
    """Argmax-decode a [L, 20] logits tensor into a 1-letter AA string."""
    idx = sequence_logits.detach().argmax(dim=-1).cpu().tolist()
    return "".join(IDX_TO_AA1[i] if 0 <= i < len(IDX_TO_AA1) else "A" for i in idx)


def _parse_model_name(model_name: Optional[str]) -> "tuple[str, Optional[str]]":
    """Split ``backend:arg`` into (backend, arg). Returns lowercased backend."""
    raw = (model_name or "").strip()
    if not raw:
        return "", None
    if ":" in raw:
        backend, arg = raw.split(":", 1)
        return backend.strip().lower(), arg.strip() or None
    return raw.strip().lower(), None


def _normalize_region_mask(
    region_mask: Optional[Union[Sequence[bool], torch.Tensor]],
    length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a [length] bool tensor on ``device`` from a list/tensor mask."""
    if region_mask is None:
        return torch.ones(length, device=device, dtype=torch.bool)
    mask = torch.as_tensor(region_mask, dtype=torch.bool, device=device)
    if mask.dim() == 0:
        mask = mask.unsqueeze(0)
    if mask.shape[0] < length:
        pad = torch.zeros(length - mask.shape[0], device=device, dtype=torch.bool)
        mask = torch.cat([mask, pad], dim=0)
    elif mask.shape[0] > length:
        mask = mask[:length]
    return mask


class TCRSequencePriorScorer(ABC):
    """Abstract interface for a TCR sequence prior scorer."""

    @abstractmethod
    def soft_prior_loss(
        self,
        sequence_logits: torch.Tensor,
        region_mask: Optional[Union[Sequence[bool], torch.Tensor]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute a differentiable TCR prior loss over sequence logits.

        Args:
            sequence_logits: [L, NUM_AMINO_ACIDS] logits (theta) for the
                mutable sequence.
            region_mask: Optional [L] boolean mask indicating which positions
                should be penalized. If None, all positions are used.
            metadata: Optional dictionary with contextual information (e.g.
                step, case id, chain type, current sequence).

        Returns:
            A scalar torch.Tensor with gradients attached.
        """
        raise NotImplementedError

    @abstractmethod
    def score_hard_sequence(
        self,
        sequence: str,
        region_mask: Optional[Union[Sequence[bool], torch.Tensor]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Score a discrete candidate sequence under the TCR prior.

        Args:
            sequence: Single-letter amino acid sequence.
            region_mask: Optional [L] boolean mask indicating which positions
                should be scored. If None, all positions are used.
            metadata: Optional dictionary with contextual information.

        Returns:
            Scalar score. Higher values indicate a more TCR-like sequence.
        """
        raise NotImplementedError


class TCRSequencePriorScorerHF(TCRSequencePriorScorer):
    """
    Placeholder HuggingFace adapter for a TCR sequence prior.

    When ``model_name`` is a valid HuggingFace model identifier, the adapter
    attempts to load an ``AutoModelForSequenceClassification`` model and its
    tokenizer. If loading fails or no model is supplied, it falls back to a
    uniform-composition placeholder so the integration remains runnable.

    Args:
        model_name: HuggingFace model identifier, or one of
            ("", "none", "dummy", "null") to use the placeholder.
        gradient_weight: Weight for the soft prior loss in the gradient path.
        verifier_weight: Weight for the hard sequence score in the verifier path.
        device: Optional torch device for the HF model.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        gradient_weight: float = 0.0,
        verifier_weight: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.model_name = (model_name or "").strip().lower()
        self.gradient_weight = float(gradient_weight)
        self.verifier_weight = float(verifier_weight)
        self.device = device
        self.model = None
        self.tokenizer = None
        self._available = False
        self._fallback = False

        # Placeholder target distribution over 20 amino acids (uniform).
        # This can be replaced with empirical TCR frequencies in the future.
        self._target_dist = torch.ones(NUM_AMINO_ACIDS) / NUM_AMINO_ACIDS

        if self.model_name and self.model_name not in ("", "none", "dummy", "null"):
            try:
                self._load_model()
            except Exception as exc:
                logger.warning(
                    "TCR prior HF model '%s' load failed; using placeholder fallback: %s",
                    self.model_name,
                    exc,
                )
                self._fallback = True
        else:
            self._fallback = True

    def _load_model(self) -> None:
        """Attempt to load the HF model and tokenizer."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.device is not None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._available = True
        logger.info("TCR prior HF model loaded: %s", self.model_name)

    def _placeholder_soft_loss(
        self,
        sequence_logits: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable placeholder: KL(P_region || uniform)."""
        P = F.softmax(sequence_logits, dim=-1)  # [L, 20]
        P_region = P[region_mask]  # [N, 20]
        if P_region.numel() == 0:
            return torch.tensor(0.0, device=sequence_logits.device, requires_grad=True)
        target = self._target_dist.to(P_region.device)
        kl = (P_region * (torch.log(P_region + 1e-8) - torch.log(target + 1e-8))).sum(dim=-1).mean()
        return kl * self.gradient_weight

    def _placeholder_hard_score(
        self,
        sequence: str,
        region_mask: torch.Tensor,
    ) -> float:
        """Placeholder hard score: average log-likelihood under uniform distribution."""
        if not region_mask.any():
            return 0.0
        seq_indices = torch.tensor(
            [AA_1_TO_IDX.get(aa, -1) for aa in sequence],
            dtype=torch.long,
        )
        if seq_indices.shape[0] != region_mask.shape[0]:
            seq_indices = seq_indices[: region_mask.shape[0]]
        seq_indices = seq_indices[region_mask.cpu()]
        if seq_indices.numel() == 0:
            return 0.0
        log_probs = torch.log(self._target_dist + 1e-8)
        return float(log_probs[seq_indices].mean().item()) * self.verifier_weight

    def _hf_hard_score(self, sequence: str) -> float:
        """Score a hard sequence with the loaded HF classification model."""
        if self.tokenizer is None or self.model is None:
            return 0.0
        with torch.no_grad():
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if self.device is not None:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            # Score = logit(positive class) - logit(negative class).
            # Positive class index 1 is assumed to mean "TCR-like / natural".
            score = float(logits[1].item()) - float(logits[0].item())
        return score * self.verifier_weight

    def soft_prior_loss(
        self,
        sequence_logits: torch.Tensor,
        region_mask: Optional[Union[Sequence[bool], torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.gradient_weight == 0.0:
            return torch.tensor(0.0, device=sequence_logits.device, requires_grad=True)

        region_mask = _normalize_region_mask(
            region_mask,
            sequence_logits.shape[0],
            sequence_logits.device,
        )
        if not self._available:
            return self._placeholder_soft_loss(sequence_logits, region_mask)

        # Differentiable loss with a real HF model is left as a future extension.
        # For now we warn once and use the placeholder so the integration remains
        # immediately runnable without a custom differentiable model head.
        if not self._fallback:
            logger.warning(
                "Differentiable soft TCR prior with a real HF model is not "
                "implemented in this placeholder; falling back to uniform-composition loss."
            )
            self._fallback = True
        return self._placeholder_soft_loss(sequence_logits, region_mask)

    def score_hard_sequence(
        self,
        sequence: str,
        region_mask: Optional[Union[Sequence[bool], torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        if self.verifier_weight == 0.0:
            return 0.0

        region_mask = _normalize_region_mask(region_mask, len(sequence), torch.device("cpu"))
        if not region_mask.any():
            return 0.0

        if self._available:
            score = self._hf_hard_score(sequence)
        else:
            score = self._placeholder_hard_score(sequence, region_mask)
        return score


# ======================================================================
# Pluggable real-model prior backends
# ======================================================================
#
# A backend exposes two primitives over the 20 canonical amino acids
# (in ``PROTEIN_RESIDUE_NAMES`` order):
#
#   per_position_logprobs(sequence, region_indices, metadata)
#       -> Optional[Tensor [len(region_indices), 20]]  (log-probs) or None
#   region_nll(sequence, region_indices, metadata)
#       -> float  (mean negative log-likelihood over the region; >= 0)
#
# The backend model is always frozen (eval, no_grad) — we never train or
# fine-tune it. The prior distribution is a *target* for the soft loss and a
# *reward* for the hard verifier; gradients never flow into the backend.


class TCRPriorBackend(ABC):
    """Abstract frozen prior-model backend over 20 canonical amino acids."""

    name: str = "abstract"
    supports_soft: bool = False
    # Structure-conditioned backends (e.g. LigandMPNN) cannot emit a
    # distribution without a per-candidate backbone (metadata['pdb_path']);
    # their real per-position output is therefore validated at score time
    # rather than at build time.
    requires_structure: bool = False

    def __init__(self) -> None:
        self._available = False
        self._real_weights = False
        self._missing_items: List[str] = []

    @abstractmethod
    def per_position_logprobs(
        self,
        sequence: str,
        region_indices: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """Return [len(region_indices), 20] log-probs, or None if unsupported."""
        raise NotImplementedError

    def region_nll(
        self,
        sequence: str,
        region_indices: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Mean NLL of the true residues over ``region_indices`` (default: via logprobs)."""
        logp = self.per_position_logprobs(sequence, region_indices, metadata)
        if logp is None or len(region_indices) == 0:
            return 0.0
        true_idx = torch.tensor(
            [AA_1_TO_IDX.get(sequence[i], -1) for i in region_indices],
            dtype=torch.long,
        )
        valid = true_idx >= 0
        if not bool(valid.any()):
            return 0.0
        rows = torch.arange(logp.shape[0])[valid]
        picked = logp[rows, true_idx[valid]]
        return float(-picked.mean().item())

    def status(self, probe_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a status dict describing whether this backend is truly usable.

        ``real_output`` is verified by actually running ``per_position_logprobs``.
        For structure-conditioned backends, pass ``probe_metadata`` containing a
        ``pdb_path`` to exercise the real scoring path; without it, the field is
        reported as None (unknown until score time).
        """
        real_output: Optional[bool] = False
        if self._available:
            if self.requires_structure and not (probe_metadata or {}).get("pdb_path"):
                real_output = None  # cannot validate without a structure
            else:
                try:
                    test_seq = "ACDEFGHIKLMNPQRSTVWY"
                    out = self.per_position_logprobs(
                        test_seq, list(range(len(test_seq))), probe_metadata
                    )
                    real_output = (
                        out is not None
                        and out.dim() == 2
                        and out.shape[1] == NUM_AMINO_ACIDS
                        and out.shape[0] > 0
                    )
                except Exception:
                    real_output = False
        return {
            "backend": self.name,
            "loaded": self._available,
            "real_weights": self._real_weights,
            "real_output": real_output,
            "requires_structure": self.requires_structure,
            "missing_items": list(self._missing_items),
        }


class TCRLMBackend(TCRPriorBackend):
    """
    hliulab/tcrLM masked-LM backend for CDR3 naturalness.

    Repo: https://github.com/hliulab/tcrLM (BERT-style masked LM over CDR3).
    This backend lazy-imports the *cloned repo* (it is not a pip/HF package)
    and loads the pretrained ``Pretrain`` model checkpoint.

    Required files/paths:
        - repo checkout dir (``TCRLM_HOME`` env or ``tcr_prior_model_name`` arg,
          e.g. ``tcrlm:/abs/path/to/tcrLM``). Must contain ``models/`` and
          ``source/`` packages.
        - a pretrained checkpoint (state_dict for ``models.tcrLM.Pretrain``),
          via ``tcr_prior_checkpoint`` config key or ``TCRLM_CKPT`` env var.
          NOTE: weights are NOT in the git repo (README ships them separately /
          via Docker image ``diaoxing359/tcrlm``).

    Scoring uses proper masked-marginal pseudo-log-likelihood: each region
    position is replaced with the TOKEN_MASK token, the model predicts the
    20-way distribution, and we read the true residue's log-prob.
    """

    name = "tcrlm"
    supports_soft = True
    # tcrLM is a CDR3 masked-LM with a fixed maximum sequence length.
    SEQ_MAX_LEN = 20

    def __init__(
        self,
        repo_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        import os

        self.repo_path = repo_path or os.environ.get("TCRLM_HOME")
        self.checkpoint = checkpoint or os.environ.get("TCRLM_CKPT")
        self.device = device or torch.device("cpu")
        self.model = None
        self._token_to_index: Dict[str, int] = {}
        self._mask_idx: Optional[int] = None
        self._pad_idx: Optional[int] = None
        # Permutation mapping tcrLM's 20-way output order -> bindcraft canonical order.
        self._reorder: Optional[List[int]] = None
        self._warned_len = False
        self._load()

    def _load(self) -> None:
        import os
        import sys

        if not self.repo_path or not os.path.isdir(self.repo_path):
            self._missing_items.append(
                f"TCRLM repo directory not found: {self.repo_path!r} "
                "(set TCRLM_HOME or tcr_prior_model_name='tcrlm:/path/to/tcrLM')"
            )
            logger.warning(
                "TCRLM backend unavailable: repo path %r not found "
                "(set TCRLM_HOME or tcr_prior_model_name='tcrlm:/path/to/tcrLM').",
                self.repo_path,
            )
            return
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        try:
            from models.tcrLM import Pretrain  # type: ignore
            from source.utils import token_to_index  # type: ignore
        except Exception as exc:  # pragma: no cover - import env dependent
            self._missing_items.append(f"TCRLM repo Python import failed: {exc}")
            logger.warning("TCRLM backend import failed from %s: %s", self.repo_path, exc)
            return

        self._token_to_index = dict(token_to_index)
        self._mask_idx = self._token_to_index.get("TOKEN_MASK")
        self._pad_idx = self._token_to_index.get("PADDING_MASK")
        # The Pretrain head outputs 20 logits in the order of the first 20
        # entries of ``residue_tokens`` (index i -> that AA). Build a permutation
        # that reorders those into bindcraft's PROTEIN_RESIDUE_NAMES order.
        idx_to_token = {v: k for k, v in self._token_to_index.items()}
        try:
            out_order = [idx_to_token[i] for i in range(NUM_AMINO_ACIDS)]
            self._reorder = [out_order.index(a) for a in IDX_TO_AA1]
        except (KeyError, ValueError) as exc:
            self._missing_items.append(f"TCRLM vocab/order incompatible: {exc}")
            logger.warning("TCRLM vocab order could not be mapped: %s", exc)
            return
        if self._mask_idx is None or self._pad_idx is None:
            self._missing_items.append("TCRLM vocab missing TOKEN_MASK/PADDING_MASK")
            return
        model = Pretrain()
        if self.checkpoint and os.path.isfile(self.checkpoint):
            try:
                state = torch.load(self.checkpoint, map_location="cpu")
                state = state.get("state_dict", state) if isinstance(state, dict) else state
                state = self._remap_state_dict(state, model)
                missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
                # A real checkpoint must populate (nearly) the entire model, not
                # just a couple of tensors. Require a high match ratio so a
                # prefix/format mismatch is caught instead of silently leaving
                # the transformer randomly initialized.
                n_model = len(model.state_dict())
                matched = n_model - len(missing_keys)
                if matched < 0.95 * n_model or missing_keys:
                    raise RuntimeError(
                        f"state_dict did not fully match Pretrain "
                        f"(matched {matched}/{n_model}, missing={len(missing_keys)}, "
                        f"unexpected={len(unexpected_keys)})"
                    )
                self._real_weights = True
                logger.info(
                    "TCRLM checkpoint loaded: %s (matched %d/%d tensors)",
                    self.checkpoint, matched, n_model,
                )
            except Exception as exc:
                self._missing_items.append(f"TCRLM checkpoint load failed: {exc}")
                logger.warning("TCRLM checkpoint load failed (%s); using random init.", exc)
        else:
            missing = (
                f"TCRLM checkpoint not found: {self.checkpoint!r} "
                "(set TCRLM_CKPT or --tcr_prior_checkpoint)"
            )
            self._missing_items.append(missing)
            logger.warning(
                "TCRLM checkpoint not provided/found (%r); scores will be meaningless "
                "until TCRLM_CKPT / tcr_prior_checkpoint points to real weights.",
                self.checkpoint,
            )
        model.eval().to(self.device)
        self.model = model
        self._available = True
        logger.info("TCRLM backend ready (repo=%s, device=%s).", self.repo_path, self.device)

    @staticmethod
    def _remap_state_dict(state: Dict[str, Any], model: "torch.nn.Module") -> Dict[str, Any]:
        """Align a checkpoint's keys to the ``Pretrain`` module hierarchy.

        The distributed tcrLM checkpoint stores the transformer at the top level
        (``token_emb.*``, ``layers.*``) plus ``fc_out.*``, whereas ``Pretrain``
        nests the transformer under ``self.protflash``. Strip any ``module.``
        (DataParallel) prefix and prepend ``protflash.`` to non-head tensors so
        they land on the right submodule.
        """
        model_keys = set(model.state_dict().keys())
        remapped: Dict[str, Any] = {}
        for k, v in state.items():
            key = k[len("module."):] if k.startswith("module.") else k
            if key in model_keys:
                remapped[key] = v
                continue
            if not key.startswith("protflash.") and f"protflash.{key}" in model_keys:
                remapped[f"protflash.{key}"] = v
            else:
                remapped[key] = v
        return remapped

    def _encode_padded(self, sequence: str) -> "tuple[torch.Tensor, int]":
        """Return (token_ids[SEQ_MAX_LEN], true_length) padded with PADDING_MASK."""
        unk = self._token_to_index.get("<UNK>", 0)
        seq = sequence[: self.SEQ_MAX_LEN]
        ids = [self._token_to_index.get(a, unk) for a in seq]
        length = len(ids)
        if length < self.SEQ_MAX_LEN:
            ids = ids + [self._pad_idx] * (self.SEQ_MAX_LEN - length)
        return torch.tensor(ids, dtype=torch.long, device=self.device), length

    def per_position_logprobs(
        self,
        sequence: str,
        region_indices: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        if not self._available or self.model is None or self._mask_idx is None:
            return None
        if len(region_indices) == 0:
            return torch.zeros(0, NUM_AMINO_ACIDS)
        if len(sequence) > self.SEQ_MAX_LEN and not self._warned_len:
            logger.warning(
                "TCRLM is a CDR3 model with max length %d; input of length %d is "
                "truncated. Pass the CDR3 region (tcr_prior_region='cdr').",
                self.SEQ_MAX_LEN, len(sequence),
            )
            self._warned_len = True
        tokens, length = self._encode_padded(sequence)
        length_t = torch.tensor([length], dtype=torch.long)
        reorder = torch.tensor(self._reorder, dtype=torch.long)
        out = torch.zeros(len(region_indices), NUM_AMINO_ACIDS)
        with torch.no_grad():
            for row, pos in enumerate(region_indices):
                if pos < 0 or pos >= length:
                    # Outside the model's window; leave as uniform log-probs.
                    out[row] = torch.full((NUM_AMINO_ACIDS,), -float(np.log(NUM_AMINO_ACIDS)))
                    continue
                masked = tokens.clone()
                masked[pos] = self._mask_idx
                logits = self.model(masked.unsqueeze(0), length_t)  # [1, SEQ_MAX_LEN, 20]
                logp = F.log_softmax(logits[0, pos, :NUM_AMINO_ACIDS].float(), dim=-1)
                out[row] = logp.index_select(0, reorder).cpu()
        return out


class ESMCBackend(TCRPriorBackend):
    """
    ESM-C backend for whole-TCR naturalness, via one of two loaders:

    - ``esm`` package (EvolutionaryScale SDK): ``model_id`` is a short name
      (``esmc_300m``, ``esmc_600m``, ``esmc_6b``); weights auto-download from
      the HuggingFace hub on first use. Selected via
      ``tcr_prior_model_name='esmc'`` or ``'esmc:esmc_600m'``.
    - ``transformers`` (``ESMCForMaskedLM``): ``model_id`` is a full HF repo
      id containing ``/`` (e.g. ``biohub/ESMC-6B``), selected via
      ``tcr_prior_model_name='esmc:biohub/ESMC-6B'``. Useful when a site only
      has ``transformers``' ESM-C support installed/cached (no ``esm``
      package, or a different repo mirror than the EvolutionaryScale one).

    By default a single forward pass yields per-position distributions
    p(a_i | full sequence) (fast, differentiable target for the soft prior).
    Set ``tcr_prior_masked=True`` in the config for exact masked-marginal
    pseudo-log-likelihood (L forward passes; slower, verifier-only recommended).
    """

    name = "esmc"
    supports_soft = True

    def __init__(
        self,
        model_id: str = "esmc_300m",
        device: Optional[torch.device] = None,
        masked: bool = False,
    ):
        super().__init__()
        self.model_id = model_id or "esmc_300m"
        self.device = device or torch.device("cpu")
        self.masked = bool(masked)
        self.client = None
        self._tokenizer = None
        self._backend_kind: Optional[str] = None  # "esm" or "transformers"
        self._vocab: Dict[str, int] = {}
        self._canon_ids: Optional[List[int]] = None
        self._load()

    def _load(self) -> None:
        if "/" in self.model_id:
            self._load_via_transformers()
        else:
            self._load_via_esm_package()

    def _load_via_esm_package(self) -> None:
        try:
            from esm.models.esmc import ESMC  # type: ignore
            from esm.tokenization import get_esmc_model_tokenizers  # type: ignore
        except Exception as exc:  # pragma: no cover - import env dependent
            self._missing_items.append(f"'esm' package not importable: {exc} (pip install esm)")
            logger.warning(
                "ESM-C backend unavailable (pip install esm): %s", exc
            )
            return
        try:
            self.client = ESMC.from_pretrained(self.model_id).to(self.device)
            self.client.eval()
            self._backend_kind = "esm"
            self._vocab = dict(get_esmc_model_tokenizers().get_vocab())
            self._canon_ids = [self._vocab.get(a, -1) for a in IDX_TO_AA1]
            self._available = True
            self._real_weights = True
            logger.info("ESM-C backend ready via 'esm' package (model=%s, device=%s).", self.model_id, self.device)
        except Exception as exc:  # pragma: no cover - network/weights dependent
            self._missing_items.append(
                f"ESM-C weights failed to load for {self.model_id!r}: {exc}"
            )
            logger.warning("ESM-C load failed for %s: %s", self.model_id, exc)

    def _load_via_transformers(self) -> None:
        try:
            from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM  # type: ignore
            from transformers.models.esmc.tokenization_esmc import ESMCTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - import env dependent
            self._missing_items.append(
                f"'transformers' ESM-C support not importable: {exc} "
                "(needs a transformers build with ESM-C support)"
            )
            logger.warning("ESM-C backend (transformers) unavailable: %s", exc)
            return
        try:
            self.client = ESMCForMaskedLM.from_pretrained(
                self.model_id, dtype=torch.float32
            ).to(self.device)
            self.client.eval()
            self._tokenizer = ESMCTokenizer()
            self._backend_kind = "transformers"
            self._vocab = dict(self._tokenizer.vocab)
            self._canon_ids = [self._vocab.get(a, -1) for a in IDX_TO_AA1]
            self._available = True
            self._real_weights = True
            logger.info(
                "ESM-C backend ready via 'transformers' (model=%s, device=%s).",
                self.model_id, self.device,
            )
        except Exception as exc:  # pragma: no cover - network/weights dependent
            self._missing_items.append(
                f"ESM-C (transformers) weights failed to load for {self.model_id!r}: {exc}"
            )
            logger.warning("ESM-C (transformers) load failed for %s: %s", self.model_id, exc)

    def _sequence_logits(
        self, sequence: str, mask_position: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Return raw per-position logits over the ESM-C vocab, shape [L, V].

        ``mask_position`` (0-indexed into ``sequence``) replaces that position's
        token with the mask token before the forward pass (masked-marginal mode).
        """
        if self._backend_kind == "transformers":
            return self._sequence_logits_transformers(sequence, mask_position)
        return self._sequence_logits_esm(sequence, mask_position)

    def _sequence_logits_esm(
        self, sequence: str, mask_position: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        from esm.sdk.api import ESMProtein, LogitsConfig  # type: ignore

        if mask_position is not None:
            sequence = sequence[:mask_position] + "_" + sequence[mask_position + 1 :]
        with torch.no_grad():
            protein = ESMProtein(sequence=sequence)
            protein_tensor = self.client.encode(protein)
            out = self.client.logits(protein_tensor, LogitsConfig(sequence=True))
            logits = out.logits.sequence  # [1, L+2, V] (BOS/EOS)
        return logits[0, 1 : 1 + len(sequence)].float().cpu()

    def _sequence_logits_transformers(
        self, sequence: str, mask_position: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        with torch.no_grad():
            enc = self._tokenizer(sequence, return_tensors="pt")
            input_ids = enc["input_ids"]
            if mask_position is not None:
                input_ids = input_ids.clone()
                input_ids[0, mask_position + 1] = self.client.config.mask_token_id  # +1 for leading <cls>
            out = self.client(
                input_ids=input_ids.to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            logits = out.logits  # [1, L+2, V] (<cls>/<eos>)
        return logits[0, 1 : 1 + len(sequence)].float().cpu()

    def _canon_logprobs(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        """Map [.., V] vocab logits to [.., 20] canonical-AA log-probs."""
        canon = torch.tensor(self._canon_ids, dtype=torch.long)
        selected = vocab_logits.index_select(-1, canon)  # [.., 20]
        return F.log_softmax(selected, dim=-1)

    def per_position_logprobs(
        self,
        sequence: str,
        region_indices: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        if not self._available or self.client is None:
            return None
        if len(region_indices) == 0:
            return torch.zeros(0, NUM_AMINO_ACIDS)

        if not self.masked:
            vocab_logits = self._sequence_logits(sequence)
            if vocab_logits is None:
                return None
            logp_all = self._canon_logprobs(vocab_logits)  # [L, 20]
            idx = torch.tensor(region_indices, dtype=torch.long)
            return logp_all.index_select(0, idx)

        # Masked-marginal mode: replace each region position with the mask token.
        out = torch.zeros(len(region_indices), NUM_AMINO_ACIDS)
        for row, pos in enumerate(region_indices):
            vocab_logits = self._sequence_logits(sequence, mask_position=pos)
            if vocab_logits is None:
                return None
            out[row] = self._canon_logprobs(vocab_logits[pos])
        return out


class LigandMPNNBackend(TCRPriorBackend):
    """
    dauparas/LigandMPNN backend for conditional CDR redesign priors.

    Repo: https://github.com/dauparas/LigandMPNN. Structure-conditioned model
    that outputs per-position amino-acid ``log_probs`` [L, 21] given a backbone
    (optionally conditioned on ligand/pMHC atoms) — suitable as a CDR prior.

    Required files/paths:
        - repo checkout dir (``LIGANDMPNN_HOME`` env or
          ``tcr_prior_model_name='ligandmpnn:/abs/path/to/LigandMPNN'``).
        - model params (``bash get_model_params.sh ./model_params``); checkpoint
          via ``tcr_prior_checkpoint`` or ``LIGANDMPNN_CKPT`` env var.
        - a per-candidate backbone PDB (pMHC + TCR) supplied at call time via
          ``metadata['pdb_path']``. Without a structure this backend is a no-op
          (returns None / 0.0), so the main pipeline is never broken.

    This is intentionally structure-driven; the ProteoCraft verifier produces
    coordinates per candidate, so wiring ``metadata['pdb_path']`` (or an
    atom array dumped to a temp PDB) enables conditional scoring.
    """

    name = "ligandmpnn"
    supports_soft = True
    requires_structure = True

    def __init__(
        self,
        repo_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_side_chain_context: bool = False,
    ):
        super().__init__()
        import os

        self.repo_path = repo_path or os.environ.get("LIGANDMPNN_HOME")
        self.checkpoint = checkpoint or os.environ.get("LIGANDMPNN_CKPT")
        self.device = device or torch.device("cpu")
        self.use_side_chain_context = bool(
            use_side_chain_context
            or os.environ.get("LIGANDMPNN_USE_SIDE_CHAIN_CONTEXT", "0") == "1"
        )
        self.model = None
        self.model_type = "protein_mpnn"
        self.atom_context_num = 1
        self.ligand_mpnn_use_side_chain_context = 0
        self._warned_no_structure = False
        self._load()

    def _load(self) -> None:
        import os
        import sys

        if not self.repo_path or not os.path.isdir(self.repo_path):
            self._missing_items.append(
                f"LigandMPNN repo directory not found: {self.repo_path!r} "
                "(set LIGANDMPNN_HOME or tcr_prior_model_name='ligandmpnn:/path')"
            )
            logger.warning(
                "LigandMPNN backend unavailable: repo path %r not found "
                "(set LIGANDMPNN_HOME or tcr_prior_model_name='ligandmpnn:/path').",
                self.repo_path,
            )
            return
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        try:
            from model_utils import ProteinMPNN  # type: ignore
        except Exception as exc:  # pragma: no cover - import env dependent
            self._missing_items.append(f"LigandMPNN repo import failed: {exc}")
            logger.warning("LigandMPNN import failed from %s: %s", self.repo_path, exc)
            return
        if not (self.checkpoint and os.path.isfile(self.checkpoint)):
            self._missing_items.append(
                f"LigandMPNN checkpoint not found: {self.checkpoint!r} "
                "(run get_model_params.sh; set LIGANDMPNN_CKPT / --tcr_prior_checkpoint)"
            )
            logger.warning(
                "LigandMPNN checkpoint not found (%r); set LIGANDMPNN_CKPT / "
                "tcr_prior_checkpoint to a ./model_params/*.pt file.",
                self.checkpoint,
            )
            return
        try:
            checkpoint = torch.load(self.checkpoint, map_location=self.device)
            # Real dauparas/LigandMPNN checkpoints (``ligandmpnn_v_*.pt``) carry an
            # ``atom_context_num`` key used to build the ligand/side-chain atom
            # k-NN context; plain ProteinMPNN checkpoints do not have this key.
            # Mirror ``run.py``'s branch exactly so a true ligand_mpnn checkpoint
            # actually runs in ligand_mpnn mode (was previously hardcoded to
            # protein_mpnn for *every* checkpoint, silently discarding ligand
            # atom context even when a real ligandmpnn_v_*.pt was supplied).
            if "atom_context_num" in checkpoint:
                model_type = "ligand_mpnn"
                atom_context_num = checkpoint["atom_context_num"]
                ligand_mpnn_use_side_chain_context = self.use_side_chain_context
            else:
                model_type = "protein_mpnn"
                atom_context_num = 1
                ligand_mpnn_use_side_chain_context = 0
            model = ProteinMPNN(
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                k_neighbors=checkpoint["num_edges"],
                device=self.device,
                atom_context_num=atom_context_num,
                model_type=model_type,
                ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device).eval()
            self.model = model
            self.model_type = model_type
            self.atom_context_num = atom_context_num
            self.ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context
            self._available = True
            self._real_weights = True
            logger.info(
                "LigandMPNN backend ready (repo=%s, ckpt=%s, model_type=%s, "
                "atom_context_num=%s). Requires metadata['pdb_path'] per "
                "candidate to score.",
                self.repo_path,
                self.checkpoint,
                model_type,
                atom_context_num,
            )
        except Exception as exc:  # pragma: no cover - weights/env dependent
            self._missing_items.append(f"LigandMPNN model construction failed: {exc}")
            logger.warning("LigandMPNN model build failed: %s", exc)

    def _score_from_pdb(
        self, pdb_path: str, region_indices: List[int]
    ) -> Optional[torch.Tensor]:
        """Run LigandMPNN scoring on a backbone (+ optional ligand/hetero atoms) PDB.

        Uses the repo's ``parse_PDB`` + ``featurize`` + ``model.score`` path to
        obtain per-position ``log_probs`` [L, 21]; we drop the 21st (X) column
        and re-normalize over the 20 canonical AAs, then select ``region_indices``.

        Mirrors ``run.py``'s ``model_type``-dependent branch: with a real
        ``ligand_mpnn`` checkpoint this parses non-protein/non-water hetero
        atoms (``other_atoms``, e.g. bound peptide/ion/glycan atoms not carried
        as standard residues) and feeds up to ``atom_context_num`` of them as
        extra structural context; with a plain ``protein_mpnn`` checkpoint it
        falls back to backbone-only conditioning (previous behavior).
        """
        try:
            import numpy as _np
            from data_utils import parse_PDB, featurize, restype_str_to_int  # type: ignore

            protein_dict, _bb, _oth, _ic, _ = parse_PDB(
                pdb_path,
                device=self.device,
                parse_all_atoms=bool(self.ligand_mpnn_use_side_chain_context),
            )
            protein_dict["chain_mask"] = torch.ones_like(protein_dict["R_idx"])
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=8.0,
                use_atom_context=1 if self.model_type == "ligand_mpnn" else 0,
                number_of_ligand_atoms=self.atom_context_num,
                model_type=self.model_type,
            )
            feature_dict["batch_size"] = 1
            L = feature_dict["S"].shape[1]
            feature_dict["randn"] = torch.randn([1, L], device=self.device)
            feature_dict["symmetry_residues"] = [[]]
            feature_dict["symmetry_weights"] = [[]]
            with torch.no_grad():
                out = self.model.score(feature_dict, use_sequence=True)
            log_probs = out["log_probs"][0].float().cpu()  # [L, 21] in LigandMPNN AA order
            # Reorder LigandMPNN's alphabetical AA columns into bindcraft's
            # canonical PROTEIN_RESIDUE_NAMES order, then renormalize over 20 AAs.
            perm = torch.tensor(
                [restype_str_to_int[a] for a in IDX_TO_AA1], dtype=torch.long
            )
            canon = log_probs.index_select(-1, perm)  # [L, 20]
            canon = F.log_softmax(canon, dim=-1)
            valid = [i for i in region_indices if 0 <= i < canon.shape[0]]
            if not valid:
                return None
            idx = torch.tensor(valid, dtype=torch.long)
            return canon.index_select(0, idx)
        except Exception as exc:
            logger.warning("LigandMPNN _score_from_pdb failed: %s", exc)
            return None

    def per_position_logprobs(
        self,
        sequence: str,
        region_indices: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        if not self._available:
            return None
        pdb_path = (metadata or {}).get("pdb_path")
        if not pdb_path:
            if not self._warned_no_structure:
                logger.warning(
                    "LigandMPNN backend needs metadata['pdb_path']; none provided — "
                    "returning no-op (prior contributes 0)."
                )
                self._warned_no_structure = True
            return None
        return self._score_from_pdb(pdb_path, region_indices)


class BackendPriorScorer(TCRSequencePriorScorer):
    """
    Prior scorer that delegates to a frozen :class:`TCRPriorBackend`.

    Soft (gradient) path:  L_soft = CE(softmax(theta_region), p_prior) * grad_w
    Hard (verifier) path:  score  = -region_nll * verifier_w  (higher = better)

    If the backend cannot produce a distribution (e.g. missing weights or
    structure), both paths degrade to a no-op so the main loop is unaffected.
    """

    def __init__(
        self,
        backend: TCRPriorBackend,
        gradient_weight: float = 0.0,
        verifier_weight: float = 0.0,
        model_name: str = "",
        strict: bool = False,
    ):
        self.backend = backend
        self.gradient_weight = float(gradient_weight)
        self.verifier_weight = float(verifier_weight)
        self.model_name = model_name or getattr(backend, "name", "backend")
        self.strict = bool(strict)
        self._warned_soft = False

    def soft_prior_loss(
        self,
        sequence_logits: torch.Tensor,
        region_mask: Optional[Union[Sequence[bool], torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        zero = torch.zeros((), device=sequence_logits.device, requires_grad=True)
        if self.gradient_weight == 0.0:
            return zero
        L = sequence_logits.shape[0]
        region_indices = _region_indices(region_mask, L)
        if not region_indices:
            return zero

        context_seq = (metadata or {}).get("current_seq") or _logits_to_sequence(sequence_logits)
        if len(context_seq) < L:
            context_seq = context_seq + "A" * (L - len(context_seq))

        logp = self.backend.per_position_logprobs(context_seq, region_indices, metadata)
        if logp is None:
            if self.strict:
                raise TCRPriorStrictError(
                    f"Strict TCR prior mode: backend '{self.model_name}' produced no "
                    "distribution for the soft prior (no-op). Missing items: "
                    f"{self.backend._missing_items or ['metadata[pdb_path] or model output']}."
                )
            if not self._warned_soft:
                logger.warning(
                    "TCR prior backend '%s' produced no distribution; soft prior is a no-op.",
                    self.model_name,
                )
                self._warned_soft = True
            return zero

        p_prior = logp.exp().to(sequence_logits.device).detach()  # [n, 20]
        idx = torch.tensor(region_indices, dtype=torch.long, device=sequence_logits.device)
        q_region = F.log_softmax(sequence_logits.index_select(0, idx), dim=-1)  # [n, 20]
        ce = -(p_prior * q_region).sum(dim=-1).mean()
        return ce * self.gradient_weight

    def score_hard_sequence(
        self,
        sequence: str,
        region_mask: Optional[Union[Sequence[bool], torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        if self.verifier_weight == 0.0:
            return 0.0
        region_indices = _region_indices(region_mask, len(sequence))
        if not region_indices:
            return 0.0
        if self.strict:
            logp = self.backend.per_position_logprobs(sequence, region_indices, metadata)
            if logp is None:
                raise TCRPriorStrictError(
                    f"Strict TCR prior mode: backend '{self.model_name}' produced no "
                    "distribution for the hard verifier score (no-op). Missing items: "
                    f"{self.backend._missing_items or ['metadata[pdb_path] or model output']}."
                )
        nll = self.backend.region_nll(sequence, region_indices, metadata)
        # naturalness (higher = better) = -NLL
        return float(-nll) * self.verifier_weight


def _build_backend(
    backend_name: str,
    arg: Optional[str],
    config: Dict[str, Any],
    device: Optional[torch.device],
) -> Optional[TCRPriorBackend]:
    """Instantiate a real prior backend, or None if the name is not recognized."""
    checkpoint = config.get("tcr_prior_checkpoint")
    if backend_name in _TCRLM_PREFIXES:
        return TCRLMBackend(repo_path=arg, checkpoint=checkpoint, device=device)
    if backend_name in _ESMC_PREFIXES:
        model_id = arg or "esmc_300m"
        masked = bool(config.get("tcr_prior_masked", False))
        return ESMCBackend(model_id=model_id, device=device, masked=masked)
    if backend_name in _LIGANDMPNN_PREFIXES:
        return LigandMPNNBackend(repo_path=arg, checkpoint=checkpoint, device=device)
    return None


def build_tcr_prior_scorer(
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> TCRSequencePriorScorer:
    """
    Build a TCR prior scorer from a configuration dictionary.

    Recognized keys (all optional):
        - use_tcr_prior_gradient (bool): enable soft loss
        - tcr_prior_gradient_weight (float): weight for soft loss
        - use_tcr_prior_verifier (bool): enable hard score
        - tcr_prior_verifier_weight (float): weight for hard score
        - tcr_prior_model_name (str): backend selector. Recognized real
          backends: ``tcrlm[:/repo]``, ``esmc[:model_id]``,
          ``ligandmpnn[:/repo]``. Anything else (incl. "", "none", "dummy",
          "null", or a plain HF id) uses the original HF/placeholder scorer.
        - tcr_prior_checkpoint (str): checkpoint path for tcrlm/ligandmpnn.
        - tcr_prior_masked (bool): ESM-C masked-marginal PLL (slower, exact).
        - require_real_tcr_prior (bool): STRICT mode. When True and a real
          backend (tcrlm/esmc/ligandmpnn) is requested, raise
          :class:`TCRPriorStrictError` instead of silently falling back if the
          backend fails to load, lacks real weights, or cannot produce real
          per-position output (i.e. would be a dummy/no-op).
    """
    config = config or {}
    use_grad = bool(config.get("use_tcr_prior_gradient", False))
    use_ver = bool(config.get("use_tcr_prior_verifier", False))
    grad_weight = float(config.get("tcr_prior_gradient_weight", 0.1)) if use_grad else 0.0
    ver_weight = float(config.get("tcr_prior_verifier_weight", 0.1)) if use_ver else 0.0
    model_name = config.get("tcr_prior_model_name", None)
    strict = bool(config.get("require_real_tcr_prior", False))

    backend_name, arg = _parse_model_name(model_name)
    is_real_backend_request = (
        backend_name in _TCRLM_PREFIXES
        or backend_name in _ESMC_PREFIXES
        or backend_name in _LIGANDMPNN_PREFIXES
    )
    backend = _build_backend(backend_name, arg, config, device)

    if backend is not None:
        probe_metadata = config.get("tcr_prior_probe_metadata")
        status = backend.status(probe_metadata)
        if strict:
            problems = []
            if not status["loaded"]:
                problems.append("model failed to load")
            if not status["real_weights"]:
                problems.append("real weights not loaded")
            # ``real_output is None`` means a structure-conditioned backend whose
            # output can only be validated at score time; that is enforced by
            # BackendPriorScorer (strict=True) instead of failing the build.
            if status["real_output"] is False:
                problems.append("no real per-position output (dummy/no-op)")
            if problems:
                raise TCRPriorStrictError(
                    "Strict TCR prior mode (require_real_tcr_prior=True): backend "
                    f"'{backend.name}' is not fully functional: {'; '.join(problems)}. "
                    f"Missing items: {status['missing_items'] or ['<none reported>']}."
                )
        logger.info(
            "TCR prior using real backend '%s' (grad_w=%.4f ver_w=%.4f) status=%s.",
            backend.name, grad_weight, ver_weight, status,
        )
        return BackendPriorScorer(
            backend=backend,
            gradient_weight=grad_weight,
            verifier_weight=ver_weight,
            model_name=backend.name,
            strict=strict,
        )

    if strict and is_real_backend_request:
        raise TCRPriorStrictError(
            "Strict TCR prior mode (require_real_tcr_prior=True): real backend "
            f"'{backend_name}' was requested but could not be instantiated."
        )

    # Fallback: original HF/placeholder scorer (unchanged behavior).
    return TCRSequencePriorScorerHF(
        model_name=model_name,
        gradient_weight=grad_weight,
        verifier_weight=ver_weight,
        device=device,
    )


def resolve_region_mask(
    tcr_prior_region: str,
    cdr_mask: Optional[Sequence[bool]],
    length: int,
) -> List[bool]:
    """
    Resolve the region mask for the TCR prior from the region flag.

    Args:
        tcr_prior_region: one of "all", "cdr", "none".
        cdr_mask: Optional CDR boolean mask; used when ``tcr_prior_region`` is "cdr".
        length: Sequence length (fallback when ``cdr_mask`` is unavailable).

    Returns:
        Boolean list of length ``length`` indicating which positions are in the
        prior region.
    """
    region = (tcr_prior_region or "all").strip().lower()
    if region not in VALID_TCR_PRIOR_REGIONS:
        raise ValueError(
            f"Unknown tcr_prior_region={region!r}; expected one of {VALID_TCR_PRIOR_REGIONS}"
        )
    if region == "none":
        return [False] * length
    if region == "cdr":
        if cdr_mask is not None:
            return list(cdr_mask)
        logger.warning(
            "tcr_prior_region='cdr' requested but no cdr_mask available; falling back to 'all'"
        )
        return [True] * length
    return [True] * length


def maybe_build_tcr_prior_scorer(
    config: Optional[Dict[str, Any]],
    device: Optional[torch.device] = None,
) -> Optional[TCRSequencePriorScorer]:
    """Return a scorer if any TCR prior flag is enabled, otherwise None."""
    if not config:
        return None
    use_grad = bool(config.get("use_tcr_prior_gradient", False))
    use_ver = bool(config.get("use_tcr_prior_verifier", False))
    if not use_grad and not use_ver:
        return None
    return build_tcr_prior_scorer(config, device=device)


def apply_tcr_verifier_score(
    result: Any,
    scorer: TCRSequencePriorScorer,
    region_mask: Union[Sequence[bool], torch.Tensor],
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Add a hard TCR prior score to a ZerothOrderResult-like object.

    The score is treated as a reward: it is converted to a loss term
    ``-score`` and added to ``total_loss`` and ``loss_dict['tcr_prior_hard']``.
    This makes the prior visible in verifier output and MCTS node metadata.
    """
    if scorer is None or scorer.verifier_weight == 0.0:
        return result

    score = scorer.score_hard_sequence(
        sequence=result.binder_seq,
        region_mask=region_mask,
        metadata=metadata,
    )
    # Convert reward-like score to a loss contribution.
    loss = -float(score)
    result.total_loss = float(result.total_loss) + loss
    if result.loss_dict is None:
        result.loss_dict = {}
    result.loss_dict["tcr_prior_hard"] = loss
    return result
