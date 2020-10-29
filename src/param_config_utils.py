def get_sdae_params(cfg, params):
    '''
    Return parameters for stacked denoising autoencoder
    '''
    if params:
        n_hid_layers = params['n_hid_layers']
        n_hid = list()
        n_hid.append(params['n_hid'])
        dropout = list()
        dropout.append(params['dropout'])
    else:
        n_hid_layers = cfg.run_cfg.n_hid_layers
        n_hid = cfg.run_cfg.n_hid
        dropout = cfg.run_cfg.dropout
    return n_hid_layers, n_hid, dropout

def get_ffnn_params(cfg, params):
    if params:
        n_hid_layers = params['n_hid_layers']
        n_hid_units = list()
        n_hid_units.append(params['n_hid'])
        act_fn = params['activation']
        
    else:
        n_hid_layers=cfg.run_cfg.n_hid_layers_ffnn
        n_hid_units=cfg.run_cfg.n_hid_ffnn
        act_fn = cfg.run_cfg.act_fn_ffnn
        
    return (n_hid_layers, n_hid_units, act_fn)