"""
Minimal 8YAF memory diagnostic - geometry + theoretical analysis.
"""
import torch, sys
sys.path.insert(0, '.')

AA3TO1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F',
    'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',
    'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R',
    'SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
}

# Check GPU availability
print("=== GPU Info ===")
total_capacity = 0
for i in range(8):
    try:
        props = torch.cuda.get_device_properties(i)
        total_capacity += props.total_memory / 1e9
        print(f"  cuda:{i}  {props.name}  {props.total_memory/1e9:.1f} GB")
    except:
        pass
print(f"  Total across all GPUs: {total_capacity:.1f} GB\n")

# Extract 8YAF geometry from PDB
from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)
structure = parser.get_structure("8YAF", "structures/8YAF.pdb")
chains = {}
for mdl in structure:
    for chain in mdl:
        seq1 = ""
        for r in chain:
            if r.id[0] == ' ' and r.resname in AA3TO1:
                seq1 += AA3TO1[r.resname]
        if seq1:
            chains[chain.id] = seq1

chain_lens = sorted(chains.items(), key=lambda x: len(x[1]), reverse=True)
binder_chain = chain_lens[0][0]
target_seq = "".join(s for c, s in chains.items() if c != binder_chain)
binder_seq = chains[binder_chain]
total_aa = len(binder_seq) + len(target_seq)

# N_token = number of residue tokens in Protenix
# AF3 tokenizes each residue as 1 token in the pairformer
N_token_conservative = total_aa
N_token_optimistic   = int(total_aa * 1.2)

print(f"=== 8YAF Geometry ===")
print(f"  Binder chain {binder_chain}: {len(binder_seq)} aa")
print(f"  Target chains ({len([c for c in chains if c != binder_chain])} chains): {len(target_seq)} aa")
print(f"  Total residues: {total_aa}")
print(f"  N_token estimate: {N_token_conservative} – {N_token_optimistic}")

H200_GB = 150.1
print(f"\n{'='*60}")
print(f"Strategy           | Mem/GPU | Fits on H200?")
print(f"{'='*60}")

for nt in [N_token_conservative, N_token_optimistic, N_token_conservative + 100]:
    pae_gb   = nt * nt * 32 * 4 / 1e9
    plddt_gb = nt * 50 * 4 / 1e9
    pde_gb   = nt * nt * 32 * 4 / 1e9
    dist_gb  = nt * nt * 49 * 4 / 1e9
    total_conf = pae_gb + plddt_gb + pde_gb + dist_gb

    # Baseline: no chunking, 1 GPU
    model_gb      = 3.5
    pairformer_gb = 20
    extra_gb      = 5
    total_1gpu = model_gb + pairformer_gb + total_conf + extra_gb
    fit_1gpu = "YES" if total_1gpu < H200_GB else "NO"

    # TP×4: Z and PAE sharded across 4 GPUs
    # Z=[N,N,128], PAE=[N,N,32]
    z_gb     = nt * nt * 128 * 4 / 1e9
    tp_z_gb  = z_gb / 4
    tp_pae_gb = pae_gb / 4
    tp_total = model_gb + (pairformer_gb - z_gb) + tp_z_gb + (total_conf - pae_gb) + tp_pae_gb + extra_gb
    fit_tp4  = "YES" if tp_total < H200_GB else "NO"

    # PP×4: pairformer layers split across 4 GPUs
    pp_total = model_gb + pairformer_gb/4 + total_conf + extra_gb
    fit_pp4  = "YES" if pp_total < H200_GB else "NO"

    print(f"\n  N_token={nt} (N_atom ~ {int(nt*4)}):")
    print(f"  confidence outputs = {total_conf:.1f} GB")
    print(f"  1 GPU (no chunk)  | {total_1gpu:6.1f} GB | {fit_1gpu}")
    print(f"  DDP×4             | {total_1gpu:6.1f} GB | {fit_1gpu} (same as 1 GPU)")
    print(f"  TP×4 (Z+PAE shard)| {tp_total:6.1f} GB | {fit_tp4}")
    print(f"  PP×4 (layers)     | {pp_total:6.1f} GB | {fit_pp4}")

print(f"\n{'='*60}")
print(f"Conclusion:")
print(f"  - H200 = 150.1 GB per GPU (8 GPUs available)")
print(f"  - No chunking + 1 GPU: depends on N_token")
print(f"  - DDP×4: DOES NOT help (activations still per-rank)")
print(f"  - TP×4: shards Z and PAE → correct approach for large outputs")
print(f"  - PP×4: shards pairformer layers → simpler to implement")
print(f"{'='*60}")
