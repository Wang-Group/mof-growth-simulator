import math


CORRECTION_TERM_FOR_DEPROTONATION = 10 ** (4.19 - 3.75)
EQUILIBRIUM_CONSTANT = 1.64
H2O_PURE = 55500.0
DMF_PURE = 12900.0
H2O_FORMATE_COEFFICIENT = 0.01
DMF_FORMATE_COEFFICIENT = 0.01
NUM_CARBOXYLATE_ON_LINKER = 3
SECOND_STEP_EQUIVALENTS = 0.0
PREBOUND_MODEL_CLUSTER_ONE_TO_ONE = "cluster_one_to_one"
PREBOUND_MODEL_MULTISITE_FIRST_BINDING_ONLY = "multisite_first_binding_only"


def solve_one_to_one_association(total_cluster_conc, total_linker_conc, association_constant):
    total_cluster_conc = max(float(total_cluster_conc), 0.0)
    total_linker_conc = max(float(total_linker_conc), 0.0)
    if association_constant is None:
        return 0.0

    association_constant = float(association_constant)
    if association_constant <= 0.0 or total_cluster_conc <= 0.0 or total_linker_conc <= 0.0:
        return 0.0

    low = 0.0
    high = min(total_cluster_conc, total_linker_conc)
    for _ in range(120):
        midpoint = 0.5 * (low + high)
        residual = midpoint - association_constant * (total_cluster_conc - midpoint) * (total_linker_conc - midpoint)
        if residual > 0.0:
            high = midpoint
        else:
            low = midpoint
    return 0.5 * (low + high)


def irreversible_second_step_capture(
    seed_complex_conc,
    total_cluster_conc,
    total_linker_conc,
    second_step_equivalents=SECOND_STEP_EQUIVALENTS,
):
    seed_complex_conc = max(float(seed_complex_conc), 0.0)
    total_cluster_conc = max(float(total_cluster_conc), 0.0)
    total_linker_conc = max(float(total_linker_conc), 0.0)
    second_step_equivalents = max(float(second_step_equivalents), 0.0)
    if seed_complex_conc <= 0.0 or second_step_equivalents <= 0.0:
        return 0.0

    free_cluster_after_seed = max(total_cluster_conc - seed_complex_conc, 0.0)
    free_linker_after_seed = max(total_linker_conc - seed_complex_conc, 0.0)
    return min(
        seed_complex_conc * second_step_equivalents,
        free_cluster_after_seed,
        free_linker_after_seed,
    )


def deprotonation_balance(
    linker_carboxylate_to_acid_ratio,
    capping_agent_conc,
    linker_conc,
    correction_term_for_deprotonation=CORRECTION_TERM_FOR_DEPROTONATION,
    num_carboxylate_on_linker=NUM_CARBOXYLATE_ON_LINKER,
):
    formate_term = (
        linker_carboxylate_to_acid_ratio
        * correction_term_for_deprotonation
        / (1.0 + linker_carboxylate_to_acid_ratio * correction_term_for_deprotonation)
        * capping_agent_conc
    )
    linker_term = (
        linker_carboxylate_to_acid_ratio
        / (1.0 + linker_carboxylate_to_acid_ratio)
        * linker_conc
        * num_carboxylate_on_linker
    )
    return formate_term + linker_term


def solve_linker_carboxylate_to_acid_ratio(
    dimethylamine_conc,
    capping_agent_conc,
    linker_conc,
    correction_term_for_deprotonation=CORRECTION_TERM_FOR_DEPROTONATION,
    num_carboxylate_on_linker=NUM_CARBOXYLATE_ON_LINKER,
):
    dimethylamine_conc = max(float(dimethylamine_conc), 0.0)
    if dimethylamine_conc <= 0.0:
        return 0.0

    high = 1.0
    while (
        deprotonation_balance(
            high,
            capping_agent_conc,
            linker_conc,
            correction_term_for_deprotonation=correction_term_for_deprotonation,
            num_carboxylate_on_linker=num_carboxylate_on_linker,
        )
        < dimethylamine_conc
    ):
        high *= 2.0
        if high > 1e16:
            break

    low = 0.0
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        if (
            deprotonation_balance(
                midpoint,
                capping_agent_conc,
                linker_conc,
                correction_term_for_deprotonation=correction_term_for_deprotonation,
                num_carboxylate_on_linker=num_carboxylate_on_linker,
            )
            < dimethylamine_conc
        ):
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def compute_acid_speciation(
    capping_agent_conc,
    linker_conc,
    dimethylamine_conc=0.0,
    correction_term_for_deprotonation=CORRECTION_TERM_FOR_DEPROTONATION,
    num_carboxylate_on_linker=NUM_CARBOXYLATE_ON_LINKER,
):
    capping_agent_conc = max(float(capping_agent_conc), 0.0)
    linker_conc = max(float(linker_conc), 0.0)
    linker_ratio = solve_linker_carboxylate_to_acid_ratio(
        dimethylamine_conc=dimethylamine_conc,
        capping_agent_conc=capping_agent_conc,
        linker_conc=linker_conc,
        correction_term_for_deprotonation=correction_term_for_deprotonation,
        num_carboxylate_on_linker=num_carboxylate_on_linker,
    )
    formate_ratio = linker_ratio * correction_term_for_deprotonation
    linker_carboxylic_acid_conc = (
        linker_conc * (1.0 / (1.0 + linker_ratio)) * num_carboxylate_on_linker
        if linker_conc > 0.0
        else 0.0
    )
    formic_acid_conc = (
        capping_agent_conc * (1.0 / (1.0 + formate_ratio))
        if capping_agent_conc > 0.0
        else 0.0
    )
    return {
        "linker_carboxylate_to_acid_ratio": linker_ratio,
        "formate_to_acid_ratio": formate_ratio,
        "linker_carboxylic_acid_conc": linker_carboxylic_acid_conc,
        "formic_acid_conc": formic_acid_conc,
    }


def effective_exchange_equilibrium_constant(
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
    capping_agent_conc,
    equilibrium_constant=EQUILIBRIUM_CONSTANT,
    h2o_formate_coefficient=H2O_FORMATE_COEFFICIENT,
    dmf_formate_coefficient=DMF_FORMATE_COEFFICIENT,
    h2o_pure=H2O_PURE,
    dmf_pure=DMF_PURE,
):
    capping_agent_conc = max(float(capping_agent_conc), 0.0)
    if capping_agent_conc <= 0.0:
        return 0.0

    h2o_dmf_ratio = max(float(h2o_dmf_ratio), 0.0)
    dmf_conc = dmf_pure / (1.0 + h2o_dmf_ratio)
    h2o_conc = h2o_pure * h2o_dmf_ratio / (1.0 + h2o_dmf_ratio)
    h2o_power = h2o_conc / capping_agent_conc * h2o_formate_coefficient
    dmf_power = dmf_conc / capping_agent_conc * dmf_formate_coefficient
    return float(equilibrium_constant_coefficient) * equilibrium_constant / (1.0 + h2o_power + dmf_power)


def derive_effective_association_constant_from_exchange(
    linker_conc,
    capping_agent_conc,
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
    dimethylamine_conc=0.0,
):
    total_linker_conc = max(float(linker_conc), 0.0)
    if total_linker_conc <= 0.0:
        acid_speciation = compute_acid_speciation(
            capping_agent_conc=capping_agent_conc,
            linker_conc=total_linker_conc,
            dimethylamine_conc=dimethylamine_conc,
        )
        return {
            "effective_exchange_equilibrium_constant": 0.0,
            "effective_association_constant": 0.0,
            "acid_speciation": acid_speciation,
        }

    acid_speciation = compute_acid_speciation(
        capping_agent_conc=capping_agent_conc,
        linker_conc=total_linker_conc,
        dimethylamine_conc=dimethylamine_conc,
    )
    effective_exchange_constant = effective_exchange_equilibrium_constant(
        equilibrium_constant_coefficient=equilibrium_constant_coefficient,
        h2o_dmf_ratio=h2o_dmf_ratio,
        capping_agent_conc=capping_agent_conc,
    )
    reactive_site_factor = acid_speciation["linker_carboxylic_acid_conc"] / total_linker_conc
    formic_acid_conc = acid_speciation["formic_acid_conc"]
    if reactive_site_factor <= 0.0 or formic_acid_conc <= 0.0:
        effective_association_constant = 0.0
    else:
        effective_association_constant = effective_exchange_constant * reactive_site_factor / formic_acid_conc

    return {
        "effective_exchange_equilibrium_constant": effective_exchange_constant,
        "effective_association_constant": effective_association_constant,
        "acid_speciation": acid_speciation,
    }


def compute_distorted_ligand_state(
    zr_conc,
    linker_conc,
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
    capping_agent_conc,
    zr6_percentage=1.0,
    association_constant_override=None,
    dimethylamine_conc=0.0,
    second_step_equivalents=SECOND_STEP_EQUIVALENTS,
):
    total_linker_conc = max(float(linker_conc), 0.0)
    total_zr6_conc = max(float(zr_conc), 0.0) * max(float(zr6_percentage), 0.0) / 6.0

    exchange_mapping = derive_effective_association_constant_from_exchange(
        linker_conc=total_linker_conc,
        capping_agent_conc=capping_agent_conc,
        equilibrium_constant_coefficient=equilibrium_constant_coefficient,
        h2o_dmf_ratio=h2o_dmf_ratio,
        dimethylamine_conc=dimethylamine_conc,
    )
    association_constant = (
        exchange_mapping["effective_association_constant"]
        if association_constant_override is None
        else float(association_constant_override)
    )
    distorted_ligand_conc = solve_one_to_one_association(
        total_cluster_conc=total_zr6_conc,
        total_linker_conc=total_linker_conc,
        association_constant=association_constant,
    )
    polymerized_distorted_conc = 0.0
    if second_step_equivalents > 0.0:
        polymerized_distorted_conc = irreversible_second_step_capture(
            seed_complex_conc=distorted_ligand_conc,
            total_cluster_conc=total_zr6_conc,
            total_linker_conc=total_linker_conc,
            second_step_equivalents=second_step_equivalents,
        )
    off_pathway_zr6_conc = distorted_ligand_conc + polymerized_distorted_conc
    off_pathway_linker_conc = distorted_ligand_conc + polymerized_distorted_conc
    free_zr6_conc = max(total_zr6_conc - off_pathway_zr6_conc, 0.0)
    free_linker_conc = max(total_linker_conc - off_pathway_linker_conc, 0.0)

    return {
        "model_name": PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
        "association_constant": association_constant,
        "effective_exchange_equilibrium_constant": exchange_mapping["effective_exchange_equilibrium_constant"],
        "acid_speciation": exchange_mapping["acid_speciation"],
        "second_step_equivalents": second_step_equivalents,
        "total_zr6_conc": total_zr6_conc,
        "total_linker_conc": total_linker_conc,
        "prebound_zr_bdc_conc": distorted_ligand_conc,
        "prebound_zr_bdc_fraction": (
            distorted_ligand_conc / total_linker_conc if total_linker_conc > 0.0 else 0.0
        ),
        "prebound_zr_btb_conc": distorted_ligand_conc,
        "prebound_zr_btb_fraction": (
            distorted_ligand_conc / total_linker_conc if total_linker_conc > 0.0 else 0.0
        ),
        "distorted_ligand_conc": distorted_ligand_conc,
        "polymerized_distorted_conc": polymerized_distorted_conc,
        "off_pathway_zr6_conc": off_pathway_zr6_conc,
        "off_pathway_linker_conc": off_pathway_linker_conc,
        "distorted_ligand_fraction": (
            distorted_ligand_conc / total_linker_conc if total_linker_conc > 0.0 else 0.0
        ),
        "off_pathway_linker_fraction": (
            off_pathway_linker_conc / total_linker_conc if total_linker_conc > 0.0 else 0.0
        ),
        "off_pathway_zr6_fraction": (
            off_pathway_zr6_conc / total_zr6_conc if total_zr6_conc > 0.0 else 0.0
        ),
        "free_zr6_conc": free_zr6_conc,
        "free_linker_conc": free_linker_conc,
        "free_zr6_fraction": free_zr6_conc / total_zr6_conc if total_zr6_conc > 0.0 else 0.0,
        "free_linker_fraction": (
            free_linker_conc / total_linker_conc if total_linker_conc > 0.0 else 0.0
        ),
    }


def default_prebound_state(
    zr_conc,
    linker_conc,
    zr6_percentage=1.0,
    effective_equilibrium_constant=0.0,
    second_step_equivalents=0.0,
    model_name=PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
    num_sites_on_cluster=12,
    num_sites_on_linker=NUM_CARBOXYLATE_ON_LINKER,
):
    total_zr6_conc = max(float(zr_conc), 0.0) * max(float(zr6_percentage), 0.0) / 6.0
    total_linker_conc = max(float(linker_conc), 0.0)
    return {
        "model_name": model_name,
        "association_constant": 0.0,
        "site_equilibrium_constant": 0.0,
        "effective_exchange_equilibrium_constant": effective_equilibrium_constant,
        "acid_speciation": None,
        "second_step_equivalents": second_step_equivalents,
        "num_sites_on_cluster": int(num_sites_on_cluster),
        "num_sites_on_linker": int(num_sites_on_linker),
        "total_zr6_conc": total_zr6_conc,
        "total_linker_conc": total_linker_conc,
        "prebound_zr_bdc_conc": 0.0,
        "prebound_zr_bdc_fraction": 0.0,
        "prebound_zr_btb_conc": 0.0,
        "prebound_zr_btb_fraction": 0.0,
        "prebound_zr6_cluster_conc": 0.0,
        "prebound_zr6_cluster_fraction": 0.0,
        "distorted_ligand_conc": 0.0,
        "polymerized_distorted_conc": 0.0,
        "off_pathway_zr6_conc": 0.0,
        "off_pathway_linker_conc": 0.0,
        "distorted_ligand_fraction": 0.0,
        "off_pathway_linker_fraction": 0.0,
        "off_pathway_zr6_fraction": 0.0,
        "bound_linker_conc": 0.0,
        "free_linker_conc": total_linker_conc,
        "free_linker_fraction": 1.0 if total_linker_conc > 0.0 else 0.0,
        "bound_zr_site_conc": 0.0,
        "free_zr_site_conc": total_zr6_conc * int(num_sites_on_cluster),
        "bound_zr_site_fraction": 0.0,
        "site_occupancy_probability": 0.0,
        "activity_ratio": 0.0,
        "mean_bound_linkers_per_cluster": 0.0,
        "released_capping_agent_conc": 0.0,
        "free_capping_agent_conc": 0.0,
        "free_zr6_conc": total_zr6_conc,
        "free_zr6_fraction": 1.0 if total_zr6_conc > 0.0 else 0.0,
        "growth_eligible_zr6_conc": total_zr6_conc,
        "growth_eligible_zr6_fraction": 1.0 if total_zr6_conc > 0.0 else 0.0,
        "cluster_occupancy_distribution": [],
    }


def normalize_prebound_model_name(model_name):
    normalized = (
        PREBOUND_MODEL_CLUSTER_ONE_TO_ONE
        if model_name is None
        else str(model_name).strip().lower().replace("-", "_")
    )
    aliases = {
        PREBOUND_MODEL_CLUSTER_ONE_TO_ONE: PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
        "one_to_one": PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
        "cluster_1to1": PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
        "cluster_one_to_one": PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
        PREBOUND_MODEL_MULTISITE_FIRST_BINDING_ONLY: PREBOUND_MODEL_MULTISITE_FIRST_BINDING_ONLY,
        "multisite": PREBOUND_MODEL_MULTISITE_FIRST_BINDING_ONLY,
        "multisite_first_binding": PREBOUND_MODEL_MULTISITE_FIRST_BINDING_ONLY,
        "multisite_first_binding_only": PREBOUND_MODEL_MULTISITE_FIRST_BINDING_ONLY,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported prebound chemistry model: {model_name}")
    return aliases[normalized]


def compute_prebound_chemistry_state(
    zr_conc,
    linker_conc,
    equilibrium_constant_coefficient,
    h2o_dmf_ratio,
    capping_agent_conc,
    zr6_percentage=1.0,
    association_constant_override=None,
    site_equilibrium_constant_override=None,
    dimethylamine_conc=0.0,
    second_step_equivalents=SECOND_STEP_EQUIVALENTS,
    model_name=PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
    num_sites_on_cluster=12,
    num_sites_on_linker=NUM_CARBOXYLATE_ON_LINKER,
):
    normalized_model_name = normalize_prebound_model_name(model_name)
    if normalized_model_name == PREBOUND_MODEL_MULTISITE_FIRST_BINDING_ONLY:
        from multisite_linker_exchange_model import compute_multisite_prebound_state

        return compute_multisite_prebound_state(
            zr_conc=zr_conc,
            linker_conc=linker_conc,
            equilibrium_constant_coefficient=equilibrium_constant_coefficient,
            h2o_dmf_ratio=h2o_dmf_ratio,
            capping_agent_conc=capping_agent_conc,
            zr6_percentage=zr6_percentage,
            site_equilibrium_constant_override=(
                association_constant_override
                if site_equilibrium_constant_override is None
                else site_equilibrium_constant_override
            ),
            dimethylamine_conc=dimethylamine_conc,
            num_sites_on_cluster=num_sites_on_cluster,
            num_sites_on_linker=num_sites_on_linker,
        )
    return compute_distorted_ligand_state(
        zr_conc=zr_conc,
        linker_conc=linker_conc,
        equilibrium_constant_coefficient=equilibrium_constant_coefficient,
        h2o_dmf_ratio=h2o_dmf_ratio,
        capping_agent_conc=capping_agent_conc,
        zr6_percentage=zr6_percentage,
        association_constant_override=association_constant_override,
        dimethylamine_conc=dimethylamine_conc,
        second_step_equivalents=second_step_equivalents,
    )


def compute_absolute_external_channel_activities(
    prebound_state,
    zr6_conc_for_growth=None,
    linker_conc_for_growth=None,
):
    if zr6_conc_for_growth is None:
        zr6_conc_for_growth = prebound_state.get("free_zr6_conc", 0.0)
    if linker_conc_for_growth is None:
        linker_conc_for_growth = prebound_state.get("free_linker_conc", 0.0)

    free_zr6_addition_activity = max(float(zr6_conc_for_growth), 0.0)
    free_linker_addition_activity = max(float(linker_conc_for_growth), 0.0)
    prebound_zr_btb_addition_activity = max(
        float(
            prebound_state.get(
                "prebound_zr6_cluster_conc",
                prebound_state.get(
                    "prebound_zr_btb_conc",
                    prebound_state.get("prebound_zr_bdc_conc", 0.0),
                ),
            )
        ),
        0.0,
    )
    total_external_addition_channel_activity = (
        free_zr6_addition_activity
        + free_linker_addition_activity
        + prebound_zr_btb_addition_activity
    )
    return {
        "free_zr6_addition_activity": free_zr6_addition_activity,
        "free_linker_addition_activity": free_linker_addition_activity,
        "prebound_zr_btb_addition_activity": prebound_zr_btb_addition_activity,
        "total_external_addition_channel_activity": total_external_addition_channel_activity,
        "external_channel_activity_basis": "absolute_concentrations",
        "prebound_external_channel_species": (
            "prebound_zr6_cluster_conc"
            if "prebound_zr6_cluster_conc" in prebound_state
            else "prebound_zr_btb_conc"
        ),
    }


def logistic_from_log_ratio(log_ratio):
    if log_ratio >= 0.0:
        exp_term = math.exp(-log_ratio)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(log_ratio)
    return exp_term / (1.0 + exp_term)


def zr6_cluster_add_probability(zr6_conc, linker_conc, num_carboxylate_on_linker=3):
    zr6_conc = max(float(zr6_conc), 0.0)
    linker_conc = max(float(linker_conc), 0.0)
    zr6_conc_reference = linker_conc * float(num_carboxylate_on_linker) / 12.0
    if zr6_conc <= 0.0 or zr6_conc_reference <= 0.0:
        return 0.0
    return logistic_from_log_ratio(-3.6 + math.log(zr6_conc / zr6_conc_reference))
