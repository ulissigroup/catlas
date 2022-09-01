import warnings
from catlas.prediction_steps import (
    parse_inputs,
    load_bulks_and_filter,
    load_adsorbates_and_filter,
    enumerate_surfaces_and_filter,
    enumerate_and_predict_adslabs,
    generate_outputs,
    finish_sankey_diagram,
)
from catlas.dask_utils import bag_split_individual_partitions

# Load inputs and define global vars
if __name__ == "__main__":
    """Run predictions according to input config file

    Usage (see examples in `.github/workflows/automated_screens`):
        If cluster is not running, start it:
            kubectl apply -f \
                configs/dask_cluster/dask_operator/catlas-hybrid-cluster.yml
            kubectl scale --replicas=4 daskworkergroup \
                catlas-hybrid-cluster-default-worker-group
            kubectl scale --replicas=1 daskworkergroup \
                catlas-hybrid-cluster-gpu-worker-group
        python bin/predictions.py path/to/config.yml \
            configs/dask_cluster/dask_operator/dask_connect.py

    Args:
        config (str): File path where a config is found. Example configs can be
            found in `configs/automated_screens` and
            `.github/workflows/automated_screens`
        dask_connect_script (str): script to connect to running dask cluster
    Raises:
        ValueError: The provided config is invalid.
    """
    config, dask_cluster_script, run_id, sankey = parse_inputs()
    exec(dask_cluster_script)

    filtered_catalyst_bag, sankey, bulk_num = load_bulks_and_filter(
        config,
        client,  # this variable is created during `exec(dask_cluster_script)`
        sankey,
    )

    # partition to 1 bulk per partition
    filtered_catalyst_bag = bag_split_individual_partitions(filtered_catalyst_bag)

    adsorbate_bag, sankey = load_adsorbates_and_filter(config, sankey)

    (
        surface_bag,
        num_unfiltered_slabs,
    ) = enumerate_surfaces_and_filter(config, filtered_catalyst_bag, bulk_num)

    (
        results_bag,
        adslab_atoms_bag,
        inference,
        most_recent_step,
    ) = enumerate_and_predict_adslabs(config, surface_bag, adsorbate_bag)

    num_adslabs, num_inferred, num_filtered_slabs = generate_outputs(
        config, results_bag, run_id, inference
    )

    # Make final updates to the sankey diagram and plot it

    if num_adslabs is None:
        num_adslabs = num_inferred = [0]
        warnings.warn(
            "Adslabs were enumerated but will not be counted for the Sankey diagram."
        )

    sankey = finish_sankey_diagram(
        sankey, num_unfiltered_slabs, num_adslabs, num_inferred, run_id
    )
