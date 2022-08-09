import plotly.graph_objects as go
import warnings
import numpy as np


class Sankey:
    def __init__(self, info_dict):
        """
        Initialize class to facilitate Sankey diagram construction.

        Args:
        info_dict: a dictionary of values that will be used to populate the output sankey diagram
        """
        self.info_dict = info_dict

    def update_dictionary(
        self, label: str, source: int, target: int, value: int, x: float, y
    ):
        """
        Updates the Sankey dictionary with a new edge (flow of content).

        Args:
            label: a new label to add to the labels
            source: the node from which the flow comes
            target: the node to which the flow goes
            value: the magnitude of the flow
        """
        if label is not None:
            self.info_dict["label"].append(label)
            self.info_dict["x"].append(x)
            self.info_dict["y"].append(y)
        self.info_dict["source"].append(source)
        self.info_dict["target"].append(target)
        self.info_dict["value"].append(value)

    def add_slab_info(self, num_unfiltered: int, num_filtered: int):
        """
        Updates the Sankey dictionary with slab information.

        Args:
            num_unfiltered: the number of slabs prior to filtering
            num_filtered: the number of slabs after filtering
        """
        node_idx = len(self.info_dict["label"])
        slab_idx = self.info_dict["label"].index("Slabs")
        self.update_dictionary(
            f"Filtered slabs ({num_filtered})",
            slab_idx,
            node_idx,
            num_filtered,
            0.6,
            0.4,
        )
        self.update_dictionary(
            f"Rejected slabs ({num_unfiltered - num_filtered})",
            slab_idx,
            node_idx + 1,
            num_unfiltered - num_filtered,
            1,
            "tbd",
        )
        self.update_dictionary(
            None,
            node_idx,
            self.info_dict["label"].index("Adslabs"),
            num_filtered,
            None,
            None,
        )
        self.info_dict["label"][slab_idx] = f"Slabs ({num_unfiltered})"

    def _add_inference_step(
        inference_list: list, step: int, node_num: int, node_idx: int
    ):
        if step < len(inferece_list):
            node_idx_next = len(self.info_dict["label"])
            node_num_next = inference_list[step]["counts"]
            self.update_dictionary(
                f"Inferred energies {inference_list[step]['label']} ({node_num_next})",
                node_idx,
                len(self.info_dict["label"]),
                node_num_next,
                0.6 + 0.4 * (step + 1) / len(inference_list),
                0.4,
            )
            if node_num != inference_list[step]["counts"]:
                self.update_dictionary(
                    f"Adslabs filtered out ({node_num - inference_list[step]['counts']})",
                    node_idx,
                    len(self.info_dict["label"]),
                    node_num - inference_list[step]["counts"],
                    0.6 + 0.4 * (step + 1) / len(inference_list),
                    0.6,
                )
            self._add_inference_step(
                inference_list, step + 1, node_num_next, node_idx_next
            )

    def add_adslab_info(self, num_adslabs: int, num_inference: list):
        """
        Updates the Sankey dictionary with adslab information.

        Args:
            num_adslabs: the number of adslabs
            num_inference: the number of inference calculations

        """
        if num_adslabs is None:
            warnings.warn(
                "Adslabs were not computed and therefore will not appear in the Sankey diagram"
            )
            num_adslabs = 0

        adslab_idx = self.info_dict["label"].index("Adslabs")
        self.info_dict["label"][adslab_idx] = f"Adslabs ({num_adslabs})"
        self._add_inference_step(num_inference, 0, num_adslabs, adslab_idx)

    def update_y_positions(self, use_log):
        """
        Update the y positions for all nodes at x = 1 so they are well spread out.

        Args:
            use_log: boolean value determining if the values will be log(values)

        """
        # Grab indices of those to change
        indices_to_change = [
            idx for idx, value in enumerate(self.info_dict["x"]) if value == 1
        ]
        # Alter values if log weighting will be used
        if use_log:
            vals = np.log(self.info_dict["value"])
            values = [val if val > 0 else 0.1 for val in vals]
            self.info_dict["value"] = values
        else:
            values = self.info_dict["value"]

        # Calculate new node placement and update vals
        flows_to_1 = [
            idx
            for idx, target in enumerate(self.info_dict["target"])
            if target in indices_to_change
        ]
        weight_factor = 0.8 / sum([values[idx] for idx in flows_to_1])
        y_sizes = [weight_factor * values[idx] for idx in flows_to_1]
        y_now = 0.9
        for idx, idx_set in enumerate(indices_to_change):
            self.info_dict["y"][idx_set] = y_now
            y_now -= y_sizes[idx]

    def get_sankey_diagram(self, run_id: str, use_log=True):
        """
        A function to create a pdf of the Sankey diagram.

        Args:
            run_id: unique id for the run to be used as a location for saving outputs

        Returns:
            a pdf of the sankey diagram for the run
        """

        # Make figure without nose placement
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=30,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=self.info_dict["label"],
                    ),
                    link=dict(
                        source=self.info_dict["source"],
                        target=self.info_dict["target"],
                        value=self.info_dict["value"],
                    ),
                )
            ]
        )
        fig.update_layout(
            autosize=False,
            width=1600,
            height=800,
        )
        fig.write_image(f"outputs/{run_id}/sankey.png")

        # Make a figure with node placement
        self.update_y_positions(use_log)
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=30,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=self.info_dict["label"],
                        x=self.info_dict["x"],
                        y=self.info_dict["y"],
                    ),
                    link=dict(
                        source=self.info_dict["source"],
                        target=self.info_dict["target"],
                        value=self.info_dict["value"],
                    ),
                )
            ]
        )
        fig.update_layout(
            autosize=False,
            width=1600,
            height=800,
        )
        fig.write_image(f"outputs/{run_id}/sankey_forced_nodes.png")
