import math

from distorted_ligand_model import (
    compute_acid_speciation,
    effective_exchange_equilibrium_constant,
)


NUM_SITES_ON_CLUSTER = 12
NUM_SITES_ON_LINKER = 3
MODEL_NAME = "multisite_first_binding_only"


def solve_multisite_first_binding_equilibrium(
    total_cluster_conc,
    total_linker_conc,
    site_equilibrium_constant,
    initial_capping_agent_conc,
    num_sites_on_cluster=NUM_SITES_ON_CLUSTER,
    reactive_linker_sites_per_free_linker=NUM_SITES_ON_LINKER,
    released_capping_agent_stoichiometry=1.0,
):
    total_cluster_conc = max(float(total_cluster_conc), 0.0)
    total_linker_conc = max(float(total_linker_conc), 0.0)
    site_equilibrium_constant = max(float(site_equilibrium_constant), 0.0)
    initial_capping_agent_conc = max(float(initial_capping_agent_conc), 0.0)
    num_sites_on_cluster = max(int(num_sites_on_cluster), 0)
    reactive_linker_sites_per_free_linker = max(float(reactive_linker_sites_per_free_linker), 0.0)
    released_capping_agent_stoichiometry = max(float(released_capping_agent_stoichiometry), 0.0)

    total_zr_sites_conc = total_cluster_conc * num_sites_on_cluster
    if (
        total_cluster_conc <= 0.0
        or total_linker_conc <= 0.0
        or total_zr_sites_conc <= 0.0
        or site_equilibrium_constant <= 0.0
        or reactive_linker_sites_per_free_linker <= 0.0
    ):
        return {
            "bound_linker_conc": 0.0,
            "free_linker_conc": total_linker_conc,
            "released_capping_agent_conc": 0.0,
            "free_capping_agent_conc": initial_capping_agent_conc,
            "bound_zr_site_conc": 0.0,
            "free_zr_site_conc": total_zr_sites_conc,
            "site_occupancy_probability": 0.0,
            "activity_ratio": 0.0,
        }

    upper_bound = min(total_linker_conc, total_zr_sites_conc)

    def site_activity_ratio(bound_linker_conc):
        free_linker_conc = max(total_linker_conc - bound_linker_conc, 0.0)
        free_capping_agent_conc = (
            initial_capping_agent_conc
            + released_capping_agent_stoichiometry * bound_linker_conc
        )
        free_capping_agent_conc = max(free_capping_agent_conc, 1e-30)
        return (
            site_equilibrium_constant
            * reactive_linker_sites_per_free_linker
            * free_linker_conc
            / free_capping_agent_conc
        )

    def site_occupancy_probability(bound_linker_conc):
        activity_ratio = site_activity_ratio(bound_linker_conc)
        if activity_ratio <= 0.0:
            return 0.0
        return activity_ratio / (1.0 + activity_ratio)

    low = 0.0
    high = upper_bound
    for _ in range(160):
        midpoint = 0.5 * (low + high)
        predicted_bound_linker_conc = total_zr_sites_conc * site_occupancy_probability(midpoint)
        if midpoint > predicted_bound_linker_conc:
            high = midpoint
        else:
            low = midpoint

    bound_linker_conc = 0.5 * (low + high)
    free_linker_conc = max(total_linker_conc - bound_linker_conc, 0.0)
    released_capping_agent_conc = released_capping_agent_stoichiometry * bound_linker_conc
    free_capping_agent_conc = initial_capping_agent_conc + released_capping_agent_conc
    bound_zr_site_conc = bound_linker_conc
    free_zr_site_conc = max(total_zr_sites_conc - bound_zr_site_conc, 0.0)
    activity_ratio = site_activity_ratio(bound_linker_conc)
    occupancy_probability = site_occupancy_probability(bound_linker_conc)

    return {
        "bound_linker_conc": bound_linker_conc,
        "free_linker_conc": free_linker_conc,
        "released_capping_agent_conc": released_capping_agent_conc,
        "free_capping_agent_conc": free_capping_agent_conc,
        "bound_zr_site_conc": bound_zr_site_conc,
        "free_zr_site_conc": free_zr_site_conc,
        "site_occupancy_probability": occupancy_probability,
        "activity_ratio": activity_ratio,
    }


def build_cluster_occupancy_distribution(
    total_cluster_conc,
    site_occupancy_probability,
    num_sites_on_cluster=NUM_SITES_ON_CLUSTER,
):
    total_cluster_conc = max(float(total_cluster_conc), 0.0)
    site_occupancy_probability = min(max(float(site_occupancy_probability), 0.0), 1.0)
    num_sites_on_cluster = max(int(num_sites_on_cluster), 0)

    raw_rows = []
    total_probability = 0.0
    for occupied_sites in range(num_sites_on_cluster + 1):
        cluster_fraction = (
            math.comb(num_sites_on_cluster, occupied_sites)
            * site_occupancy_probability ** occupied_sites
            * (1.0 - site_occupancy_probability) ** (num_sites_on_cluster - occupied_sites)
        )
        raw_rows.append((occupied_sites, cluster_fraction))
        total_probability += cluster_fraction

    if total_probability <= 0.0:
        return []

    rows = []
    for occupied_sites, cluster_fraction in raw_rows:
        cluster_fraction /= total_probability
        cluster_conc = total_cluster_conc * cluster_fraction
        rows.append(
            {
                "bound_linkers_per_cluster": occupied_sites,
                "cluster_fraction": cluster_fraction,
                "cluster_conc": cluster_conc,
                "bound_linker_conc": occupied_sites * cluster_conc,
                "remaining_aa_sites_per_cluster": num_sites_on_cluster - occupied_sites,
                "remaining_aa_site_conc": (num_sites_on_cluster - occupied_sites) * cluster_conc,
            }
        )
    return rows


def compute_multisite_exchange_state(
    zr_conc,
    linker_conc,
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
    capping_agent_conc,
    zr6_percentage=1.0,
    site_equilibrium_constant_override=None,
    dimethylamine_conc=0.0,
    num_sites_on_cluster=NUM_SITES_ON_CLUSTER,
    num_sites_on_linker=NUM_SITES_ON_LINKER,
):
    total_linker_conc = max(float(linker_conc), 0.0)
    total_zr6_conc = max(float(zr_conc), 0.0) * max(float(zr6_percentage), 0.0) / 6.0
    num_sites_on_cluster = max(int(num_sites_on_cluster), 0)
    num_sites_on_linker = max(int(num_sites_on_linker), 0)

    acid_speciation = compute_acid_speciation(
        capping_agent_conc=capping_agent_conc,
        linker_conc=total_linker_conc,
        dimethylamine_conc=dimethylamine_conc,
        num_carboxylate_on_linker=num_sites_on_linker,
    )
    site_equilibrium_constant = (
        effective_exchange_equilibrium_constant(
            equilibrium_constant_coefficient=equilibrium_constant_coefficient,
            h2o_dmf_ratio=h2o_dmf_ratio,
            capping_agent_conc=capping_agent_conc,
        )
        if site_equilibrium_constant_override is None
        else max(float(site_equilibrium_constant_override), 0.0)
    )

    reactive_linker_sites_per_free_linker = (
        acid_speciation["linker_carboxylic_acid_conc"] / total_linker_conc
        if total_linker_conc > 0.0
        else 0.0
    )
    equilibrium_solution = solve_multisite_first_binding_equilibrium(
        total_cluster_conc=total_zr6_conc,
        total_linker_conc=total_linker_conc,
        site_equilibrium_constant=site_equilibrium_constant,
        initial_capping_agent_conc=acid_speciation["formic_acid_conc"],
        num_sites_on_cluster=num_sites_on_cluster,
        reactive_linker_sites_per_free_linker=reactive_linker_sites_per_free_linker,
    )
    occupancy_distribution = build_cluster_occupancy_distribution(
        total_cluster_conc=total_zr6_conc,
        site_occupancy_probability=equilibrium_solution["site_occupancy_probability"],
        num_sites_on_cluster=num_sites_on_cluster,
    )

    total_zr_site_conc = total_zr6_conc * num_sites_on_cluster
    bound_linker_conc = equilibrium_solution["bound_linker_conc"]
    mean_bound_linkers_per_cluster = (
        bound_linker_conc / total_zr6_conc if total_zr6_conc > 0.0 else 0.0
    )

    return {
        "model_name": MODEL_NAME,
        "assumptions": [
            "Treat K_eff as a site-level exchange constant for one Zr-AA site exchanging with one linker.",
            "Each Zr6 cluster has a fixed number of equivalent exchangeable sites.",
            "A free BTB linker contributes three equivalent first-binding sites by default.",
            "Once a linker binds through one end, the remaining linker ends are not allowed to react further in this model.",
            "Released AA is added back to the solution reservoir with 1:1 stoichiometry.",
        ],
        "site_equilibrium_constant": site_equilibrium_constant,
        "acid_speciation": acid_speciation,
        "reactive_linker_sites_per_free_linker": reactive_linker_sites_per_free_linker,
        "num_sites_on_cluster": num_sites_on_cluster,
        "num_sites_on_linker": num_sites_on_linker,
        "total_zr6_conc": total_zr6_conc,
        "total_linker_conc": total_linker_conc,
        "total_zr_site_conc": total_zr_site_conc,
        "bound_linker_conc": bound_linker_conc,
        "free_linker_conc": equilibrium_solution["free_linker_conc"],
        "released_capping_agent_conc": equilibrium_solution["released_capping_agent_conc"],
        "free_capping_agent_conc": equilibrium_solution["free_capping_agent_conc"],
        "bound_zr_site_conc": equilibrium_solution["bound_zr_site_conc"],
        "free_zr_site_conc": equilibrium_solution["free_zr_site_conc"],
        "site_occupancy_probability": equilibrium_solution["site_occupancy_probability"],
        "activity_ratio": equilibrium_solution["activity_ratio"],
        "mean_bound_linkers_per_cluster": mean_bound_linkers_per_cluster,
        "bound_linker_fraction_of_molecules": (
            bound_linker_conc / total_linker_conc if total_linker_conc > 0.0 else 0.0
        ),
        "free_linker_fraction_of_molecules": (
            equilibrium_solution["free_linker_conc"] / total_linker_conc
            if total_linker_conc > 0.0
            else 0.0
        ),
        "bound_zr_site_fraction": (
            equilibrium_solution["bound_zr_site_conc"] / total_zr_site_conc
            if total_zr_site_conc > 0.0
            else 0.0
        ),
        "cluster_occupancy_distribution": occupancy_distribution,
    }


def compute_multisite_prebound_state(
    zr_conc,
    linker_conc,
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
    capping_agent_conc,
    zr6_percentage=1.0,
    site_equilibrium_constant_override=None,
    dimethylamine_conc=0.0,
    num_sites_on_cluster=NUM_SITES_ON_CLUSTER,
    num_sites_on_linker=NUM_SITES_ON_LINKER,
):
    raw_state = compute_multisite_exchange_state(
        zr_conc=zr_conc,
        linker_conc=linker_conc,
        equilibrium_constant_coefficient=equilibrium_constant_coefficient,
        h2o_dmf_ratio=h2o_dmf_ratio,
        capping_agent_conc=capping_agent_conc,
        zr6_percentage=zr6_percentage,
        site_equilibrium_constant_override=site_equilibrium_constant_override,
        dimethylamine_conc=dimethylamine_conc,
        num_sites_on_cluster=num_sites_on_cluster,
        num_sites_on_linker=num_sites_on_linker,
    )

    total_zr6_conc = raw_state["total_zr6_conc"]
    total_linker_conc = raw_state["total_linker_conc"]
    free_zr6_conc = (
        raw_state["free_zr_site_conc"] / raw_state["num_sites_on_cluster"]
        if raw_state["num_sites_on_cluster"] > 0
        else 0.0
    )
    free_zr6_fraction = (
        free_zr6_conc / total_zr6_conc
        if total_zr6_conc > 0.0
        else 0.0
    )
    occupancy_distribution = raw_state["cluster_occupancy_distribution"]
    growth_eligible_zr6_conc = sum(
        row["cluster_conc"]
        for row in occupancy_distribution
        if row["remaining_aa_sites_per_cluster"] > 0
    )
    growth_eligible_zr6_fraction = (
        growth_eligible_zr6_conc / total_zr6_conc
        if total_zr6_conc > 0.0
        else 0.0
    )
    prebound_zr6_conc = sum(
        row["cluster_conc"]
        for row in occupancy_distribution
        if row["bound_linkers_per_cluster"] > 0
    )
    prebound_zr6_fraction = (
        prebound_zr6_conc / total_zr6_conc
        if total_zr6_conc > 0.0
        else 0.0
    )

    return {
        "model_name": raw_state["model_name"],
        "assumptions": raw_state["assumptions"],
        "association_constant": raw_state["site_equilibrium_constant"],
        "site_equilibrium_constant": raw_state["site_equilibrium_constant"],
        "effective_exchange_equilibrium_constant": raw_state["site_equilibrium_constant"],
        "acid_speciation": raw_state["acid_speciation"],
        "second_step_equivalents": 0.0,
        "num_sites_on_cluster": raw_state["num_sites_on_cluster"],
        "num_sites_on_linker": raw_state["num_sites_on_linker"],
        "reactive_linker_sites_per_free_linker": raw_state["reactive_linker_sites_per_free_linker"],
        "total_zr6_conc": total_zr6_conc,
        "total_linker_conc": total_linker_conc,
        "prebound_zr_bdc_conc": raw_state["bound_linker_conc"],
        "prebound_zr_bdc_fraction": raw_state["bound_linker_fraction_of_molecules"],
        "prebound_zr_btb_conc": raw_state["bound_linker_conc"],
        "prebound_zr_btb_fraction": raw_state["bound_linker_fraction_of_molecules"],
        "prebound_zr6_cluster_conc": prebound_zr6_conc,
        "prebound_zr6_cluster_fraction": prebound_zr6_fraction,
        "distorted_ligand_conc": raw_state["bound_linker_conc"],
        "polymerized_distorted_conc": 0.0,
        "off_pathway_zr6_conc": 0.0,
        "off_pathway_linker_conc": 0.0,
        "distorted_ligand_fraction": raw_state["bound_linker_fraction_of_molecules"],
        "off_pathway_linker_fraction": 0.0,
        "off_pathway_zr6_fraction": 0.0,
        "bound_linker_conc": raw_state["bound_linker_conc"],
        "free_linker_conc": raw_state["free_linker_conc"],
        "free_linker_fraction": raw_state["free_linker_fraction_of_molecules"],
        "bound_zr_site_conc": raw_state["bound_zr_site_conc"],
        "free_zr_site_conc": raw_state["free_zr_site_conc"],
        "bound_zr_site_fraction": raw_state["bound_zr_site_fraction"],
        "site_occupancy_probability": raw_state["site_occupancy_probability"],
        "activity_ratio": raw_state["activity_ratio"],
        "mean_bound_linkers_per_cluster": raw_state["mean_bound_linkers_per_cluster"],
        "released_capping_agent_conc": raw_state["released_capping_agent_conc"],
        "free_capping_agent_conc": raw_state["free_capping_agent_conc"],
        "free_zr6_conc": free_zr6_conc,
        "free_zr6_fraction": free_zr6_fraction,
        "growth_eligible_zr6_conc": growth_eligible_zr6_conc,
        "growth_eligible_zr6_fraction": growth_eligible_zr6_fraction,
        "cluster_occupancy_distribution": occupancy_distribution,
    }
