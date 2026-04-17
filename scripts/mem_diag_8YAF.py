"""
8YAF memory diagnostic - determine N_token and OOM stage precisely.
"""
import sys, torch
sys.path.insert(0, '.')

device = torch.device("cuda:3")
torch.cuda.set_device(device)
torch.cuda.empty_cache()

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from protenix.model.protenix import Protenix
from bindcraft.utils import build_complex_features, prepare_input_features

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict): d[k] = deep_update(d.get(k, {}), v)
        else: d[k] = v
    return d

model_name = "protenix_base_default_v1.0.0"
ckpt_dir = "checkpoint"
base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
deep_update(base_configs, model_configs[model_name])
arg_str = (f"--model_name {model_name} --use_msa false "
           f"--dtype fp32 --load_checkpoint_dir {ckpt_dir} "
           f"--dump_dir /tmp/bindcraft_hallucination")
configs = parse_configs(configs=base_configs, arg_str=arg_str, fill_required_with_null=True)

print("Loading model...")
model = Protenix(configs).to(device)
ckpt = torch.load(f"{ckpt_dir}/protenix_base_default_v1.0.0.pt", map_location=device, weights_only=False)
state_dict = ckpt["model"]
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()
torch.cuda.reset_peak_memory_stats()

def m(tag):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated(device) / 1e9
    peak  = torch.cuda.max_memory_allocated(device) / 1e9
    cap   = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"  [{tag:32s}] alloc={alloc:.2f} GB  peak={peak:.2f} GB  cap={cap:.1f} GB")

# Extract 8YAF sequences from PDB
from Bio.PDB import PDBParser
AA3TO1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
          'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
          'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
parser = PDBParser(QUIET=True)
structure = parser.get_structure("8YAF", "structures/8YAF.pdb")
chains = {}
for mdl in structure:
    for chain in mdl:
        seq1 = "".join(AA3TO1[r.resname] for r in chain if r.id[0]==' ' and r.resname in AA3TO1)
        if seq1: chains[chain.id] = seq1
chain_lens = sorted(chains.items(), key=lambda x: len(x[1]), reverse=True)
binder_chain = chain_lens[0][0]
target_seq = "".join(s for c,s in chains.items() if c != binder_chain)
binder_seq = chains[binder_chain]

print(f"\n8YAF: binder={binder_chain}({len(binder_seq)}aa) + target({len(target_seq)}aa)")

print("Building features...")
input_dict, N_token, N_atom = build_complex_features(binder_seq, target_seq, device)
input_dict = prepare_input_features(model, input_dict)

pae_gb = N_token**2 * 32 * 4 / 1e9
pde_gb = N_token**2 * 32 * 4 / 1e9
dist_gb = N_token**2 * 49 * 4 / 1e9
z_gb    = N_token**2 * 128 * 4 / 1e9
total_conf = pae_gb + N_token * 50 * 4 / 1e9 + pde_gb + dist_gb

print(f"  N_token={N_token}, N_atom={N_atom}")
print(f"  Z pair embedding  = [{N_token}²×128] × 4B = {z_gb:.2f} GB")
print(f"  PAE logits        = [{N_token}²×32]  × 4B = {pae_gb:.2f} GB")
print(f"  Total confidence  = {total_conf:.2f} GB")
m("model loaded")

# Test A: pairformer no chunk
print("\n=== A: pairformer (no chunk) ===")
torch.cuda.reset_peak_memory_stats(device)
try:
    with torch.no_grad():
        s_inputs, s_trunk, z_trunk = model.get_pairformer_output(
            input_dict, N_cycle=4, inplace_safe=False, chunk_size=None)
    m("pairformer(chunk=None) SUCCESS")
    pairformer_ok = True
except RuntimeError as e:
    print(f"  FAILED: {e}")
    pairformer_ok = False
    torch.cuda.reset_peak_memory_stats(device)

# Test B: confidence_head no chunk (correct usage: x_pred from diffusion, shape [N_sample, N_atom, 3])
print("\n=== B: confidence_head (no chunk) ===")
torch.cuda.reset_peak_memory_stats(device)
try:
    with torch.no_grad():
        # Get coordinate from diffusion (returns [N_sample=1, N_atom, 3])
        coordinate = model.sample_diffusion(
            denoise_net=model.diffusion_module,
            input_feature_dict=input_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=None,
            p_lm=None,
            c_l=None,
            N_sample=1,
            noise_schedule=model.inference_noise_scheduler(N_step=model.configs.sample_diffusion["N_step"], device=device, dtype=s_inputs.dtype),
            inplace_safe=True,
            enable_efficient_fusion=model.enable_efficient_fusion,
        )  # [1, N_atom, 3]
        plddt_logits, pae_logits, pde_logits, dist_logits = model.run_confidence_head(
            input_feature_dict=input_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
            pair_mask=None, x_pred_coords=coordinate,
            triangle_multiplicative=model.configs.triangle_multiplicative,
            triangle_attention=model.configs.triangle_attention,
            inplace_safe=False, chunk_size=None,
        )
    m("confidence_head(chunk=None) SUCCESS")
    del plddt_logits, pae_logits, pde_logits, dist_logits, coordinate
    torch.cuda.empty_cache()
    confidence_ok = True
except RuntimeError as e:
    print(f"  FAILED: {e}")
    confidence_ok = False
    torch.cuda.reset_peak_memory_stats(device)

# Test C: zeroth_order no chunk (full forward via _run_full_forward)
print("\n=== C: zeroth_order (no chunk) ===")
torch.cuda.reset_peak_memory_stats(device)
try:
    from bindcraft.hallucination.zeroth_order import _run_full_forward
    coord, s_in, s_tr, z_tr, plddt, pae, pde, dist, contacts = _run_full_forward(
        model, input_dict, device, N_cycle=4, chunk_size=None)
    m("zeroth_order(chunk=None) SUCCESS")
    zero_ok = True
except RuntimeError as e:
    print(f"  FAILED: {e}")
    zero_ok = False
    torch.cuda.reset_peak_memory_stats(device)

# Test D: zeroth_order with chunk=256 (for comparison)
print("\n=== D: zeroth_order (chunk=256) ===")
torch.cuda.reset_peak_memory_stats(device)
try:
    from bindcraft.hallucination.zeroth_order import _run_full_forward
    coord, s_in, s_tr, z_tr, plddt, pae, pde, dist, contacts = _run_full_forward(
        model, input_dict, device, N_cycle=4, chunk_size=256)
    m("zeroth_order(chunk=256) SUCCESS")
    chunk_ok = True
except RuntimeError as e:
    print(f"  FAILED: {e}")
    chunk_ok = False
    torch.cuda.reset_peak_memory_stats(device)

cap = torch.cuda.get_device_properties(device).total_memory / 1e9
print(f"\n{'='*55}")
print(f"H200 usable memory: {cap:.1f} GB")
print(f"Pairformer (no chunk):   {'PASS' if pairformer_ok else 'FAIL'}")
print(f"Confidence (no chunk):   {'PASS' if confidence_ok else 'FAIL'}")
print(f"Zeroth order (no chunk): {'PASS' if zero_ok else 'FAIL'}")
print(f"Zeroth order (chunk=256):{'PASS' if chunk_ok else 'FAIL'}")
print(f"{'='*55}")
