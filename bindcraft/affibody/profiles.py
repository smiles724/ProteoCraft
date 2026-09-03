"""Canonical LibA/LibB scaffold definitions and sequence validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def normalize_amino_acids(values: Iterable[str] | str) -> tuple[str, ...]:
    """Return a deterministic, unique amino-acid tuple.

    Strings may be compact (``ADEFI``) or comma/space separated.
    """

    if isinstance(values, str):
        text = values.upper().replace(",", " ")
        items = list("".join(text.split()))
    else:
        items = [str(value).strip().upper() for value in values]
    invalid = sorted({aa for aa in items if len(aa) != 1 or aa not in STANDARD_AAS})
    if invalid:
        raise ValueError(f"invalid amino acid(s): {invalid}")
    unique = tuple(dict.fromkeys(items))
    if not unique:
        raise ValueError("amino-acid alphabet cannot be empty")
    return unique


@dataclass(frozen=True)
class AffibodyProfile:
    """A fixed Affibody framework plus an experimentally sampled interface."""

    name: str
    scaffold_template: str
    mutable_positions_1based: tuple[int, ...]
    allowed_aas: tuple[str, ...]
    default_tuple: str
    assay_time_min: int

    def __post_init__(self) -> None:
        template = self.scaffold_template.strip().upper()
        positions = tuple(int(position) for position in self.mutable_positions_1based)
        allowed = normalize_amino_acids(self.allowed_aas)
        if not template:
            raise ValueError("scaffold_template cannot be empty")
        if any(aa not in STANDARD_AAS and aa != "X" for aa in template):
            raise ValueError("scaffold_template may only contain standard amino acids and X")
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("mutable positions must be non-empty and unique")
        if any(position < 1 or position > len(template) for position in positions):
            raise ValueError("mutable position outside scaffold sequence")
        x_positions = tuple(index + 1 for index, aa in enumerate(template) if aa == "X")
        if x_positions != positions:
            raise ValueError(
                "X positions in scaffold_template must exactly match mutable positions: "
                f"{x_positions} != {positions}"
            )
        default_tuple = self.default_tuple.strip().upper()
        if len(default_tuple) != len(positions):
            raise ValueError("default_tuple length must match mutable position count")
        if any(aa not in allowed for aa in default_tuple):
            raise ValueError("default_tuple contains an amino acid outside allowed_aas")
        object.__setattr__(self, "scaffold_template", template)
        object.__setattr__(self, "mutable_positions_1based", positions)
        object.__setattr__(self, "allowed_aas", allowed)
        object.__setattr__(self, "default_tuple", default_tuple)

    @property
    def length(self) -> int:
        return len(self.scaffold_template)

    @property
    def mutable_mask(self) -> tuple[bool, ...]:
        positions = set(self.mutable_positions_1based)
        return tuple(index in positions for index in range(1, self.length + 1))

    @property
    def mutable_mask_argument(self) -> str:
        return "".join("1" if value else "0" for value in self.mutable_mask)

    @property
    def allowed_aas_argument(self) -> str:
        return "".join(self.allowed_aas)

    def render(self, variable_tuple: str | None = None) -> str:
        """Render a full sequence by replacing the template X positions."""

        variable_tuple = (variable_tuple or self.default_tuple).strip().upper()
        if len(variable_tuple) != len(self.mutable_positions_1based):
            raise ValueError(
                f"{self.name} tuple must have length {len(self.mutable_positions_1based)}"
            )
        invalid = sorted({aa for aa in variable_tuple if aa not in self.allowed_aas})
        if invalid:
            raise ValueError(
                f"{self.name} tuple contains residues outside its experimental "
                f"alphabet: {invalid}"
            )
        chars = list(self.scaffold_template)
        for position, aa in zip(self.mutable_positions_1based, variable_tuple):
            chars[position - 1] = aa
        return "".join(chars)

    def extract_tuple(self, sequence: str) -> str:
        sequence = self.validate_sequence(sequence)
        return "".join(sequence[position - 1] for position in self.mutable_positions_1based)

    def validate_sequence(self, sequence: str) -> str:
        """Validate framework preservation and the experimental alphabet."""

        sequence = str(sequence).strip().upper()
        if len(sequence) != self.length:
            raise ValueError(
                f"{self.name} sequence length {len(sequence)} != expected {self.length}"
            )
        invalid = sorted(set(sequence) - STANDARD_AAS)
        if invalid:
            raise ValueError(f"sequence contains invalid amino acid(s): {invalid}")
        mutable = set(self.mutable_positions_1based)
        for index, (template_aa, observed_aa) in enumerate(
            zip(self.scaffold_template, sequence), start=1
        ):
            if index in mutable:
                if observed_aa not in self.allowed_aas:
                    raise ValueError(
                        f"{self.name} position {index} residue {observed_aa} is outside "
                        "the experimental alphabet"
                    )
            elif observed_aa != template_aa:
                raise ValueError(
                    f"{self.name} framework position {index} changed from "
                    f"{template_aa} to {observed_aa}"
                )
        return sequence

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scaffold_template": self.scaffold_template,
            "mutable_positions_1based": list(self.mutable_positions_1based),
            "allowed_aas": "".join(self.allowed_aas),
            "default_tuple": self.default_tuple,
            "default_sequence": self.render(),
            "binder_length": self.length,
            "mutable_mask": self.mutable_mask_argument,
            "assay_time_min": self.assay_time_min,
        }


AFFIBODY_PROFILES: Mapping[str, AffibodyProfile] = {
    "liba": AffibodyProfile(
        name="LibA",
        scaffold_template=(
            "VDNKNKFNKEQQNAXYEIXHLPNLNEEQXNAFXQSLKDDPSQSANLLAEAKKLNDAQAPK"
        ),
        mutable_positions_1based=(15, 19, 29, 33),
        allowed_aas=tuple("ADEFHIKLNPQSTVY"),
        default_tuple="DSHK",
        assay_time_min=15,
    ),
    "libb": AffibodyProfile(
        name="LibB",
        scaffold_template=(
            "VDNKNKFXKEFXNAXXEIXHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAPK"
        ),
        mutable_positions_1based=(8, 12, 15, 16, 19),
        allowed_aas=tuple("ADEFIKLMNSTVY"),
        default_tuple="NNYYF",
        assay_time_min=30,
    ),
}


def get_affibody_profile(name: str) -> AffibodyProfile:
    key = str(name).strip().lower()
    try:
        return AFFIBODY_PROFILES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown Affibody profile {name!r}; choose from {sorted(AFFIBODY_PROFILES)}"
        ) from exc
