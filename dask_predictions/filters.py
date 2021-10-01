import warnings
import numpy as np

all_bulk_filters = [
    'mpids',
    'elements',
    'number_elements',
    'object_size',
    'Pourbaix_stability'
]

all_surf_filters = [
    
]

def filterparse_config(config):
   
    filters = {name:val for name,val in config.items() if (len(name) >= 10) and (name[:9] == 'filter_by') and (config[name]['enable'])}
    
    bulk_filters = {name:val for name,val in filters.items() if name[10:] in all_bulk_filters}
    surf_filters = {name:val for name,val in filters.items() if name[10:] in all_surf_filters}
    
    unknown_filters = {name:val for name,val in filters.items() if (name not in all_bulk_filters) and (name not in all_surf_filters)}
    if len(unknown_filters)>0:
        warnings.warn('Filters could not be parsed: ' + str([c for c in unknown_filters.keys()]))
        
    return bulk_filters, surf_filters
    
        
def bulk_filter(config, dask_df):
    bulk_filters, _ = filterparse_config(config)
    
    for name, val in bulk_filters.items():
        if name == 'filter_by_mpids':
            dask_df = dask_df[dask_df.bulk_mpid.isin(config['filter_by_mpids']['mpid_list'])]
        elif name == 'filter_by_Pourbaix_stability':
            pass
        elif name == 'filter_by_elements':
            dask_df = dask_df[dask_df.bulk_elements.apply(lambda x: all([el in config['filter_by_elements']['element_list'] for el in x]))]
        elif name == 'filter_by_num_elements':
            dask_df = dask_df[dask_df.bulk_elements.apply(len).isin(config['filter_by_number_elements']['number_of_els'])]
        elif name == 'filter_by_object_size':
            dask_df = dask_df[dask_df.bulk_natoms <= config['filter_by_object_size']['max_size']]
        else:
            warnings.warn('Filter is not implemented: ' + name)
    return dask_df