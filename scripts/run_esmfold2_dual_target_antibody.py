"""Explicit masked-antibody entry point; no CDR boundaries are inferred."""
from run_esmfold2_dual_target import main

if __name__ == "__main__":
    main(required_design_mode="masked_antibody")
