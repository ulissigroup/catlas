from parity_utils import get_predicted_E, data_preprocessing, apply_filters, get_specific_smile_plot, get_general_plot
import yaml
import sys

if __name__ == "__main__":

    # Load the config yaml
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    
    # Load the data
    df = data_preprocessing(config['npz_file_path'], config['dft_data_path'])
    
    # Apply filters
    df_filtered = apply_filters(config, df)
    
    # Generate adsorbate specific plots
    for smile in config['desired_adsorbate_smiles']:
        get_specific_smile_plot(smile, df_filtered, config['npz_file_path'])
        
    # Generate overall model plot
    get_general_plot(df_filtered, config['npz_file_path'])
   