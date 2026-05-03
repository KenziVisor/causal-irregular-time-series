import argparse
import os
import pickle
import sys
from pathlib import Path

if "--validate-config-only" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/physionet2012_causal_graph.py",
        fixed_dataset="physionet",
    )

import matplotlib.pyplot as plt
import networkx as nx


DEFAULT_GRAPH_PKL_PATH = "../data/causal_graph.pkl"
DEFAULT_GRAPH_PNG_PATH = "../PhysioNet 2012 – Causal DAG.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the PhysioNet 2012 causal DAG and save the graph artifacts."
    )
    parser.add_argument(
        "--dataset-config-csv",
        default=None,
        help=(
            "Path to the dataset global-variables CSV. If omitted, use the default "
            "PhysioNet config."
        ),
    )
    parser.add_argument(
        "--graph-pkl-path",
        default=None,
        help=f"Output path for the graph pickle. Default: {DEFAULT_GRAPH_PKL_PATH}",
    )
    parser.add_argument(
        "--graph-png-path",
        default=None,
        help=f"Output path for the rendered graph PNG. Default: {DEFAULT_GRAPH_PNG_PATH}",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without creating graph outputs.",
    )
    return parser.parse_args()


def resolve_output_path(path_like: str) -> str:
    raw_path = path_like.strip()
    if not raw_path:
        raise ValueError("Output path must be a non-empty string.")
    return os.path.abspath(os.path.expanduser(raw_path))


def create_physionet2012_causal_graph(
    save=0,
    graph_pkl_path: str | None = None,
) -> nx.DiGraph:
    """
    Create a directed acyclic graph (DAG) representing the latent-variable
    causal structure for PhysioNet / CinC Challenge 2012 variables.

    Returns
    -------
    nx.DiGraph
        A directed graph where nodes are variables (background + latent +
        observed/process/outcome nodes) and edges represent assumed causal
        relationships.
    """

    G = nx.DiGraph()

    # ------------------------------------------------------------------
    # Background variables
    # ------------------------------------------------------------------
    background_vars = [
        "BG_Age",
        "BG_Gender",
        "BG_HeightWeightBMI",
        "BG_ICUType",
    ]

    # ------------------------------------------------------------------
    # Latent variables
    # ------------------------------------------------------------------
    latent_vars = [
        "LAT_CHRONIC_BASELINE_RISK",
        "LAT_INFLAMMATION_SEPSIS_BURDEN",
        "LAT_GLOBAL_SEVERITY",
        "LAT_SHOCK",
        "LAT_RESPIRATORY_FAILURE",
        "LAT_RENAL_DYSFUNCTION",
        "LAT_HEPATIC_DYSFUNCTION",
        "LAT_COAG_HEME_DYSFUNCTION",
        "LAT_NEUROLOGIC_DYSFUNCTION",
        "LAT_CARDIAC_INJURY_STRAIN",
        "LAT_METABOLIC_DERANGEMENT",
    ]

    # ------------------------------------------------------------------
    # Observed measurement, treatment/care-process, missingness, and outcome nodes
    # ------------------------------------------------------------------
    observed_vars = [
        "OBS_TempWBCInflam",
        "OBS_TroponinHR",
        "OBS_RespiratoryGasExchange",
        "OBS_Hemodynamics",
        "OBS_RenalLabsUrine",
        "OBS_LiverLabs",
        "OBS_CBCPlatelets",
        "OBS_GCS",
        "OBS_MetabolicLabsABG",
        "OBS_AvailabilityCounts",
        "TRT_MechanicalVentilation",
        "MISS_LactateABGOrdering",
        "MISS_TroponinOrdering",
        "MISS_MeasurementIntensity",
        "OUT_InHospitalMortality",
    ]

    # ------------------------------------------------------------------
    # Add all nodes
    # ------------------------------------------------------------------
    G.add_nodes_from(background_vars, node_type="background")
    G.add_nodes_from(latent_vars, node_type="latent")
    G.add_nodes_from(observed_vars, node_type="observed")

    # ------------------------------------------------------------------
    # Final latent causal DAG edge list
    # ------------------------------------------------------------------
    edges = [
        ("BG_Age", "LAT_CHRONIC_BASELINE_RISK"),
        ("BG_Gender", "LAT_CHRONIC_BASELINE_RISK"),
        ("BG_HeightWeightBMI", "LAT_CHRONIC_BASELINE_RISK"),
        ("BG_ICUType", "LAT_CHRONIC_BASELINE_RISK"),
        ("BG_ICUType", "LAT_CARDIAC_INJURY_STRAIN"),
        ("BG_ICUType", "MISS_MeasurementIntensity"),

        ("LAT_CHRONIC_BASELINE_RISK", "LAT_GLOBAL_SEVERITY"),
        ("LAT_CHRONIC_BASELINE_RISK", "OUT_InHospitalMortality"),

        ("LAT_INFLAMMATION_SEPSIS_BURDEN", "LAT_GLOBAL_SEVERITY"),
        ("LAT_INFLAMMATION_SEPSIS_BURDEN", "LAT_SHOCK"),
        ("LAT_INFLAMMATION_SEPSIS_BURDEN", "LAT_COAG_HEME_DYSFUNCTION"),
        ("LAT_INFLAMMATION_SEPSIS_BURDEN", "OBS_TempWBCInflam"),

        ("LAT_CARDIAC_INJURY_STRAIN", "LAT_SHOCK"),
        ("LAT_CARDIAC_INJURY_STRAIN", "OUT_InHospitalMortality"),
        ("LAT_CARDIAC_INJURY_STRAIN", "OBS_TroponinHR"),
        ("LAT_CARDIAC_INJURY_STRAIN", "MISS_TroponinOrdering"),

        ("LAT_RESPIRATORY_FAILURE", "LAT_METABOLIC_DERANGEMENT"),
        ("LAT_RESPIRATORY_FAILURE", "TRT_MechanicalVentilation"),
        ("LAT_RESPIRATORY_FAILURE", "OUT_InHospitalMortality"),
        ("LAT_RESPIRATORY_FAILURE", "OBS_RespiratoryGasExchange"),

        ("LAT_SHOCK", "LAT_RENAL_DYSFUNCTION"),
        ("LAT_SHOCK", "LAT_HEPATIC_DYSFUNCTION"),
        ("LAT_SHOCK", "LAT_METABOLIC_DERANGEMENT"),
        ("LAT_SHOCK", "OUT_InHospitalMortality"),
        ("LAT_SHOCK", "OBS_Hemodynamics"),
        ("LAT_SHOCK", "MISS_LactateABGOrdering"),

        ("LAT_RENAL_DYSFUNCTION", "LAT_METABOLIC_DERANGEMENT"),
        ("LAT_RENAL_DYSFUNCTION", "OUT_InHospitalMortality"),
        ("LAT_RENAL_DYSFUNCTION", "OBS_RenalLabsUrine"),

        ("LAT_HEPATIC_DYSFUNCTION", "LAT_COAG_HEME_DYSFUNCTION"),
        ("LAT_HEPATIC_DYSFUNCTION", "OBS_LiverLabs"),

        ("LAT_COAG_HEME_DYSFUNCTION", "OUT_InHospitalMortality"),
        ("LAT_COAG_HEME_DYSFUNCTION", "OBS_CBCPlatelets"),

        ("LAT_NEUROLOGIC_DYSFUNCTION", "OUT_InHospitalMortality"),
        ("LAT_NEUROLOGIC_DYSFUNCTION", "OBS_GCS"),

        ("LAT_METABOLIC_DERANGEMENT", "OUT_InHospitalMortality"),
        ("LAT_METABOLIC_DERANGEMENT", "OBS_MetabolicLabsABG"),

        ("LAT_GLOBAL_SEVERITY", "OUT_InHospitalMortality"),
        ("LAT_GLOBAL_SEVERITY", "TRT_MechanicalVentilation"),
        ("LAT_GLOBAL_SEVERITY", "MISS_MeasurementIntensity"),

        ("TRT_MechanicalVentilation", "OBS_RespiratoryGasExchange"),
        ("MISS_MeasurementIntensity", "OBS_AvailabilityCounts"),
        ("MISS_LactateABGOrdering", "OBS_MetabolicLabsABG"),
        ("MISS_LactateABGOrdering", "OBS_RespiratoryGasExchange"),
        ("MISS_TroponinOrdering", "OBS_TroponinHR"),
    ]
    G.add_edges_from(edges)

    # ------------------------------------------------------------------
    # Sanity check: ensure DAG
    # ------------------------------------------------------------------
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Constructed graph is not a DAG")

    print(
        f"      Built PhysioNet DAG with {G.number_of_nodes()} nodes and "
        f"{G.number_of_edges()} edges."
    )

    if save:
        filename_graph = resolve_output_path(
            graph_pkl_path or DEFAULT_GRAPH_PKL_PATH
        )
        os.makedirs(os.path.dirname(filename_graph), exist_ok=True)
        print(f"      Saving graph pickle to: {filename_graph}")
        with open(filename_graph, "wb") as file:
            pickle.dump(G, file)
        if os.path.exists(filename_graph):
            size = os.path.getsize(filename_graph)
            if size > 0:
                print(f"      Saved graph pickle ({size} bytes).")
    return G

def draw_graph(
    G: nx.DiGraph,
    save = 0,
    graph_png_path: str | None = None,
    figsize=(22, 18),
    node_size=1400,
    font_size=8,
):
    """
    Draw a hierarchical causal DAG with visible directed arrows.
    Node colors:
      - background : light blue
      - latent     : orange
      - observed   : light green
    """

    # ------------------------------------------------------------
    # Color mapping
    # ------------------------------------------------------------
    color_map = {
        "background": "#9ecae1",  # light blue
        "latent": "#fdae6b",      # orange
        "observed": "#a1d99b",    # light green
    }

    # ------------------------------------------------------------
    # Group nodes
    # ------------------------------------------------------------
    background_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "background"
    ]
    latent_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "latent"
    ]
    observed_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "observed"
    ]

    # ------------------------------------------------------------
    # Hierarchical layout (manual & deterministic)
    # ------------------------------------------------------------
    pos = {}

    def _assign_layer(nodes, y, x_spacing):
        x_offset = -(len(nodes) - 1) * x_spacing / 2
        for i, node in enumerate(nodes):
            pos[node] = (x_offset + i * x_spacing, y)

    _assign_layer(background_nodes, y=3.0, x_spacing=2.2)
    _assign_layer(latent_nodes, y=2.0, x_spacing=2.0)
    _assign_layer(observed_nodes, y=1.0, x_spacing=0.9)

    # ------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------
    plt.figure(figsize=figsize)

    # --- EDGES (with real arrows) ---
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=16,
        edge_color="gray",
        width=1.2,
        alpha=0.7,
        connectionstyle="arc3,rad=0.05",
        min_source_margin=10,
        min_target_margin=15,
    )

    # --- NODES ---
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=background_nodes,
        node_color=color_map["background"],
        node_size=node_size,
        edgecolors="black",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=latent_nodes,
        node_color=color_map["latent"],
        node_size=node_size,
        edgecolors="black",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=observed_nodes,
        node_color=color_map["observed"],
        node_size=node_size,
        edgecolors="black",
    )

    # --- LABELS ---
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
    )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=12,
        )
        for label, color in color_map.items()
    ]

    plt.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    plt.title("PhysioNet 2012 – Causal DAG", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    if save:
        file_name = resolve_output_path(graph_png_path or DEFAULT_GRAPH_PNG_PATH)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        print(f"      Saving DAG figure to: {file_name}")
        plt.savefig(file_name)
        if os.path.exists(file_name):
            size = os.path.getsize(file_name)
            if size > 0:
                print(f"      Saved DAG figure ({size} bytes).")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    graph_pkl_path = args.graph_pkl_path or DEFAULT_GRAPH_PKL_PATH
    graph_png_path = args.graph_png_path or DEFAULT_GRAPH_PNG_PATH
    print("=== Building PhysioNet 2012 causal DAG ===")
    print("[1/2] Creating graph structure")
    g = create_physionet2012_causal_graph(
        save=1,
        graph_pkl_path=graph_pkl_path,
    )
    print("[2/2] Rendering graph figure")
    draw_graph(
        g,
        save=1,
        graph_png_path=graph_png_path,
    )
    print("PhysioNet 2012 causal DAG build completed.")


if __name__ == "__main__":
    main()
