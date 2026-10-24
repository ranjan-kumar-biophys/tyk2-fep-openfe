"""Step 1: charge the ligands and plan the relative binding free energy network.

Writes work/network/:
  ligands_charged.sdf          AM1-BCC charges, computed once and reused by every edge
  ligand_network.graphml       the planned network (open with `openfe view-ligand-network`)
  edges.csv                    ligand pairs and their LOMAP scores
  transformations/*.json       one per leg (complex + solvent) of every edge, for `openfe quickrun`
plus figures/network.png. --smoke writes a single tiny solvent-leg transformation to
work/network_smoke/ to check the toolchain on a laptop in a few minutes.
"""
import argparse
import functools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import yaml
from openff.units import unit
from rdkit import Chem

import gufe
import openfe
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges
from openfe.setup import ligand_network_planning as lnp
from openfe.setup.alchemical_network_planner import RBFEAlchemicalNetworkPlanner

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "inputs" / "benchmark"


def protocol_settings(cfg, smoke):
    p = cfg["protocol"]
    s = RelativeHybridTopologyProtocol.default_settings()
    s.thermo_settings.temperature = cfg["temperature_K"] * unit.kelvin
    s.protocol_repeats = p["repeats"]
    s.lambda_settings.lambda_windows = p["lambda_windows"]
    s.simulation_settings.n_replicas = p["lambda_windows"]
    s.simulation_settings.equilibration_length = p["equilibration_ns"] * unit.nanosecond
    s.simulation_settings.production_length = p["production_ns"] * unit.nanosecond
    if smoke:  # a few picoseconds: proves the pipeline runs, nothing more
        s.protocol_repeats = 1
        s.simulation_settings.minimization_steps = 100
        s.simulation_settings.equilibration_length = 10 * unit.picosecond
        s.simulation_settings.production_length = 20 * unit.picosecond
        s.simulation_settings.real_time_analysis_interval = None
        s.engine_settings.compute_platform = None  # fastest available, e.g. a laptop GPU
    return s


def load_charged_ligands(cfg, out):
    sdf = out / "ligands_charged.sdf"
    if sdf.exists():
        return [openfe.SmallMoleculeComponent(m)
                for m in Chem.SDMolSupplier(str(sdf), removeHs=False)]
    ligands = [openfe.SmallMoleculeComponent(m)
               for m in Chem.SDMolSupplier(str(BENCH / "ligands.sdf"), removeHs=False)]
    ligands = bulk_assign_partial_charges(
        ligands, overwrite=False, method=cfg["charges"]["method"], toolkit_backend="ambertools",
        generate_n_conformers=None, nagl_model=None, processors=cfg["charges"]["processors"])
    sdf.write_text("".join(l.to_sdf() for l in ligands))
    return ligands


def draw_network(ligand_network, path):
    g = nx.Graph()
    for m in ligand_network.edges:
        g.add_edge(m.componentA.name.replace("lig_", ""), m.componentB.name.replace("lig_", ""),
                   score=m.annotations.get("score", float("nan")))
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.kamada_kawai_layout(g)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#52514e", width=1.2)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#2a78d6", node_size=520,
                           edgecolors="#fcfcfb", linewidths=2)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7, font_color="#fcfcfb")
    ax.set_title(f"Perturbation network: {g.number_of_nodes()} ligands, {g.number_of_edges()} edges",
                 loc="left", fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="one tiny solvent leg for a quick local test")
    args = p.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    out = ROOT / "work" / ("network_smoke" if args.smoke else "network")
    (out / "transformations").mkdir(parents=True, exist_ok=True)
    ligands = load_charged_ligands(cfg, ROOT / "work" / "network")  # charges shared by both modes
    print(f"{len(ligands)} charged ligands")

    mapper = (openfe.KartografAtomMapper() if cfg["mapper"] == "kartograf"
              else openfe.LomapAtomMapper(time=20, element_change=False))
    planner_fn = (functools.partial(lnp.generate_minimal_redundant_network, mst_num=2)
                  if cfg["network"] == "redundant" else lnp.generate_minimal_spanning_network)
    planner = RBFEAlchemicalNetworkPlanner(
        name="tyk2", mappers=[mapper], mapping_scorer=openfe.lomap_scorers.default_lomap_score,
        ligand_network_planner=planner_fn,
        protocol=RelativeHybridTopologyProtocol(protocol_settings(cfg, args.smoke)))
    protein = openfe.ProteinComponent.from_pdb_file(str(BENCH / "protein.pdb"))
    solvent = openfe.SolventComponent()  # water + 0.15 M NaCl, neutralising
    network = planner(ligands=ligands, solvent=solvent, protein=protein)

    transformations = sorted(network.edges, key=lambda t: t.name)
    if args.smoke:
        transformations = [t for t in transformations if "solvent" in t.name][:1]
    for t in transformations:
        t.to_json(out / "transformations" / f"{t.name}.json")

    mappings = {t.mapping for t in network.edges if t.mapping is not None}
    ligand_network = gufe.LigandNetwork(edges=mappings)
    (out / "ligand_network.graphml").write_text(ligand_network.to_graphml())
    with open(out / "edges.csv", "w") as fh:
        fh.write("ligand_a,ligand_b,lomap_score\n")
        for m in sorted(mappings, key=lambda m: (m.componentA.name, m.componentB.name)):
            fh.write(f"{m.componentA.name},{m.componentB.name},{m.annotations.get('score', '')}\n")
    if not args.smoke:
        (ROOT / "figures").mkdir(exist_ok=True)
        draw_network(ligand_network, ROOT / "figures" / "network.png")

    s = cfg["protocol"]
    per_window_ns = s["equilibration_ns"] + s["production_ns"]
    complex_ns = len(mappings) * s["lambda_windows"] * per_window_ns * s["repeats"]
    summary = {"ligands": len(ligands), "edges": len(mappings),
               "transformations_written": len(transformations),
               "complex_leg_sampling_ns": complex_ns, "settings": s}
    (out / "plan_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
