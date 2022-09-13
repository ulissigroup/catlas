import yaml
from catlas.parity.parity_utils import get_parity_upfront
from catlas.config_validation import config_validator
import sys
from jinja2 import Template
import os
import time

# Load inputs and define global vars
if __name__ == "__main__":
    """Get parity plots for a model.
    Args:
        config_path (str): the path where the input config is located
    Raises:
        ValueError: The provided config is invalid.
    """
    # Load the config yaml
    config_path = sys.argv[1]
    template = Template(open(config_path).read())
    config = yaml.load(template.render(**os.environ), Loader=yaml.FullLoader)
    if not config_validator.validate(config):
        raise ValueError(
            "Config has the following errors:\n%s"
            % "\n".join(
                [
                    ": ".join(['"%s"' % str(i) for i in item])
                    for item in config_validator.errors.items()
                ]
            )
        )
    else:
        print("Config validated")
    # Establish run information
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + config["output_options"]["run_name"]
    os.makedirs(f"outputs/{run_id}/")

    # Print catlas to terminal
    with open("catlas/catlas_ascii.txt", "r") as f:
        print(f.read())

    # Generate parity plots
    if ("make_parity_plots" in config["output_options"]) and (
        config["output_options"]["make_parity_plots"]
    ):
        get_parity_upfront(config, run_id)
        print(
            """Parity plots are ready if data was available, please review them to
                ensure the model selected meets your needs."""
        )
