import os
from pathlib import Path

import yaml
from cerberus import Validator

from pymatgen.core import Element


mpid_regex = "^mp-\d+$|^mvc-\d+$"  # 'mp-#' or 'mvc-#'
valid_element_groups = [
    "transition metal",
    "post-transition metal",
    "metalloid",
    "rare earth metal",
    "alkali",
    "alkaline",
    "chalcogen",
    "halogen",
]


def validate_element(field, value, error):
    invalid_elements = [e for e in value if not Element.is_valid_symbol(e)]
    if len(invalid_elements) > 0:
        error(
            field,
            "invalid elements: [%s]"
            % ", ".join(['"%s"' % e for e in invalid_elements]),
        )


def validate_path_exists(field, value, error):
    if not os.path.exists(value):
        error(field, "file path does not exist: '%s'" % value)


def validate_folder_exists(field, value, error):
    """A more permissive check to check if a file/folder can be created if it doesn't exist"""
    path_list = value.split("/")
    value = "/".join(path_list[:-1])
    if not os.path.exists(value):
        error(field, "file path's enclosing folder does not exist: '%s'" % value)


config_schema = {
    "validate": {"type": "boolean"},
    "memory_cache_location": {"type": "string", "check_with": validate_folder_exists},
    "input_options": {
        "required": True,
        "type": "dict",
        "schema": {
            "adsorbate_file": {
                "required": True,
                "type": "string",
                "check_with": validate_path_exists,
            },
            "bulk_file": {
                "required": True,
                "type": "string",
                "check_with": validate_path_exists,
            },
        },
    },
    "adsorbate_filters": {
        "type": "dict",
        "schema": {"filter_by_smiles": {"type": "list", "schema": {"type": "string"}}},
    },
    "bulk_filters": {
        "type": "dict",
        "schema": {
            "filter_by_bulk_ids": {"type": "list", "regex": mpid_regex},
            "filter_ignore_mpids": {"type": "list", "regex": mpid_regex},
            "filter_by_acceptable_elements": {
                "type": "list",
                "check_with": validate_element,
            },
            "filter_by_required_elements": {
                "type": "list",
                "check_with": validate_element,
            },
            "filter_by_num_elements": {
                "type": "list",
                "schema": {"type": "integer", "min": 1},
            },
            "filter_by_object_size": {"type": "integer"},
            "filter_by_elements_active_host": {
                "type": "dict",
                "schema": {
                    "active": {
                        "type": "list",
                        "schema": {"type": "string", "check_with": validate_element},
                    },
                    "host": {
                        "type": "list",
                        "schema": {"type": "string", "check_with": validate_element},
                    },
                },
            },
            "filter_by_bulk_e_above_hull": {
                "type": "float",
            },
            "filter_by_bulk_band_gap": {"type": "float"},
            "filter_by_element_groups": {
                "type": "list",
                "allowed": valid_element_groups,
            },
            "filter_by_pourbaix_stability": {
                "type": "dict",
                "schema": {
                    "lmdb_path": {
                        "required": True,
                        "type": "string",
                        "check_with": validate_folder_exists,
                    },
                    "conditions_list": {
                        "required": True,
                        "excludes": [
                            "pH_lower",
                            "pH_upper",
                            "pH_step",
                            "V_lower",
                            "V_upper",
                            "V_step",
                        ],
                        "type": "list",
                        "schema": {
                            "type": "dict",
                            "required": True,
                            "schema": {
                                "pH": {"type": "float"},
                                "V": {"type": "float"},
                                "max_decomposition_energy": {"type": "float"},
                            },
                        },
                    },
                    "pH_lower": {
                        "required": True,
                        "excludes": "conditions_list",
                        "dependencies": ["pH_upper", "V_lower", "V_upper"],
                    },
                    "pH_upper": {
                        "type": "float",
                        "dependencies": "pH_lower",
                    },
                    "pH_step": {
                        "type": "float",
                        "dependencies": "pH_lower",
                    },
                    "V_lower": {
                        "type": "float",
                        "dependencies": "pH_lower",
                    },
                    "max_decomposition_energy": {
                        "type": "float",
                        "dependencies": "pH_lower",
                    },
                    "V_upper": {"type": "float", "dependencies": "pH_lower"},
                    "V_step": {"type": "float", "dependencies": "pH_lower"},
                },
            },
        },
    },
    "slab_filters": {
        "type": "dict",
        "schema": {
            "filter_by_object_size": {"type": "integer"},
            "filter_by_max_miller_index": {"type": "integer"},
        },
    },
    "output_options": {
        "required": True,
        "type": "dict",
        "schema": {
            "make_parity_plots": {"type": "boolean"},
            "output_all_structures": {"type": "boolean"},
            "pickle_intermediate_outputs": {"type": "boolean"},
            "pickle_final_output": {"type": "boolean"},
            "verbose": {"type": "boolean"},
            "run_name": {"required": True, "type": "string"},
        },
    },
    "adslab_prediction_steps": {
        "required": False,
        "type": "list",
        "schema": {
            "anyof": [
                {
                    "type": "dict",
                    "schema": {
                        "checkpoint_path": {
                            "type": "string",
                            "check_with": validate_path_exists,
                            "regex": ".*.pt",  # cerberus doesn't understand re "$"; requires full match
                        },
                        "gpu": {"type": "boolean", "required": True},
                        "label": {
                            "required": True,
                            "type": "string",
                        },
                        "number_steps": {
                            "type": "integer",
                        },
                        "batch_size": {"type": "integer"},
                        "step_type": {"allowed": ["inference"], "required": True},
                    },
                },
                {
                    "type": "dict",
                    "schema": {
                        "step_type": {
                            "allowed": ["filter_by_adsorption_energy_target"],
                            "required": True,
                        },
                        "target_value": {"type": "float", "required": True},
                        "range_value": {"type": "float"},
                        "adsorbate_smiles": {"type": "string", "required": True},
                        "hash_columns": {"type": "list", "schema": {"type": "string"}},
                        "filter_column": {"type": "string", "required": True},
                    },
                },
                {
                    "type": "dict",
                    "schema": {
                        "step_type": {
                            "allowed": ["filter_by_adsorption_energy"],
                            "required": True,
                        },
                        "min_value": {"type": "float"},
                        "max_value": {"type": "float"},
                        "adsorbate_smiles": {"type": "string", "required": True},
                        "hash_columns": {"type": "list", "schema": {"type": "string"}},
                        "filter_column": {"type": "string", "required": True},
                    },
                },
            ]
        },
    },
}

config_validator = Validator(config_schema)
