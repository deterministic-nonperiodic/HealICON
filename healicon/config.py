import yaml

def load_variable_mapping(config_path: str) -> dict:
    """
    Load a YAML configuration file containing variable mappings.
    Example YAML format:
    variables:
        t: temp        # Maps output 't' to input 'temp'
        u: u_wind      # Maps output 'u' to input 'u_wind'
    """
    if config_path is None:
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config.get('variables', {})

def apply_cf_conventions(ds):
    """
    Attempt to identify standard CF variables (e.g. temperature, wind)
    if no specific mapping is provided, or as a fallback.
    Returns a dictionary of standard_name -> variable_name.
    """
    cf_map = {}
    for var_name, da in ds.data_vars.items():
        standard_name = da.attrs.get("standard_name", None)
        if standard_name:
            cf_map[standard_name] = var_name
    return cf_map
