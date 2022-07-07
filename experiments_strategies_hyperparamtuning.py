# These are the experiments used for the hyperparameter tuning, not featured in the paper

# 1A Which covariance function is optimal for EI 0.01?
compare_covariance_functions_EI = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_ei_0.01_matern32_0.25': {
        'name': 'bayes_opt_ei_0.01_matern32_0.25',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '0.25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.25,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern32_0.5': {
        'name': 'bayes_opt_ei_0.01_matern32_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01': {
        'name': 'bayes_opt_ei_0.01',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_ei_0.01_matern32_1.5': {
        'name': 'bayes_opt_ei_0.01_matern32_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 1.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern32_2.0': {
        'name': 'bayes_opt_ei_0.01_matern32_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern32_2.5': {
        'name': 'bayes_opt_ei_0.01_matern32_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern52_0.25': {
        'name': 'bayes_opt_ei_0.01_matern52_0.25',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '0.25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 0.25,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern52_0.5': {
        'name': 'bayes_opt_ei_0.01_matern52_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 0.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern52': {
        'name': 'bayes_opt_ei_0.01_matern52',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 1.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern52_1.5': {
        'name': 'bayes_opt_ei_0.01_matern52_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 1.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern52_2.0': {
        'name': 'bayes_opt_ei_0.01_matern52_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 2.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_ei_0.01_matern52_2.5': {
        'name': 'bayes_opt_ei_0.01_matern52_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 2.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
}

# 1B Which covariance function is optimal for PoI 0.01?
compare_covariance_functions_PoI = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_poi_0.01_matern32_0.5': {
        'name': 'bayes_opt_poi_0.01_matern32_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_poi_0.01': {
        'name': 'bayes_opt_poi_0.01',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_poi_0.01_matern32_1.5': {
        'name': 'bayes_opt_poi_0.01_matern32_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 1.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_poi_0.01_matern32_2.0': {
        'name': 'bayes_opt_poi_0.01_matern32_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_poi_0.01_matern32_2.5': {
        'name': 'bayes_opt_poi_0.01_matern32_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_poi_0.01_matern52_0.5': {
        'name': 'bayes_opt_poi_0.01_matern52_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 0.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_poi_0.01_matern52': {
        'name': 'bayes_opt_poi_0.01_matern52',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 1.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_poi_0.01_matern52_1.5': {
        'name': 'bayes_opt_poi_0.01_matern52_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 1.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_poi_0.01_matern52_2.0': {
        'name': 'bayes_opt_poi_0.01_matern52_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 2.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_poi_0.01_matern52_2.5': {
        'name': 'bayes_opt_poi_0.01_matern52_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 2.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
}

# 1C Which covariance function is optimal for LCB 0.01?
compare_covariance_functions_LCB = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_lcb_0.01_matern32_0.5': {
        'name': 'bayes_opt_lcb_0.01_matern32_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_lcb_0.01': {
        'name': 'bayes_opt_lcb_0.01',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_0.01_matern32_1.5': {
        'name': 'bayes_opt_lcb_0.01_matern32_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 1.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_lcb_0.01_matern32_2.0': {
        'name': 'bayes_opt_lcb_0.01_matern32_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_lcb_0.01_matern32_2.5': {
        'name': 'bayes_opt_lcb_0.01_matern32_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 3/2',
        'bar_group': '2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_lcb_0.01_matern52_0.5': {
        'name': 'bayes_opt_lcb_0.01_matern52_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 0.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_lcb_0.01_matern52': {
        'name': 'bayes_opt_lcb_0.01_matern52',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 1.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_lcb_0.01_matern52_1.5': {
        'name': 'bayes_opt_lcb_0.01_matern52_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 1.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_lcb_0.01_matern52_2.0': {
        'name': 'bayes_opt_lcb_0.01_matern52_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 2.0,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
    'bayes_opt_lcb_0.01_matern52_2.5': {
        'name': 'bayes_opt_lcb_0.01_matern52_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'Matern 5/2',
        'bar_group': '2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern52',
            'covariancelengthscale': 2.5,
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30,
            }
        }
    },
}

# 2 Which exploration factor is optimal for Matern 3/2 lengthscale 2.0?
compare_exploration_factor_matern_32_2 = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_ei_0.01': {
        'name': 'bayes_opt_ei_0.01',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': '0.01',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_ei_0.1': {
        'name': 'bayes_opt_ei_0.1',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': '0.1',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'methodparams': {
                'explorationfactor': 0.1,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_ei_0.3': {
        'name': 'bayes_opt_ei_0.3',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': '0.3',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'methodparams': {
                'explorationfactor': 0.3,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_ei_CV': {
        'name': 'bayes_opt_ei_CV',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_poi_0.01': {
        'name': 'bayes_opt_poi_0.01',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': '0.01',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_poi_0.1': {
        'name': 'bayes_opt_poi_0.1',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': '0.1',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'methodparams': {
                'explorationfactor': 0.1,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_poi_0.3': {
        'name': 'bayes_opt_poi_0.3',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': '0.3',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'methodparams': {
                'explorationfactor': 0.3,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_poi_CV': {
        'name': 'bayes_opt_poi_CV',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_0.01': {
        'name': 'bayes_opt_lcb_0.01',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': '0.01',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_0.1': {
        'name': 'bayes_opt_lcb_0.1',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': '0.1',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'methodparams': {
                'explorationfactor': 0.1,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_0.3': {
        'name': 'bayes_opt_lcb_0.3',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': '0.3',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'methodparams': {
                'explorationfactor': 0.3,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_CV': {
        'name': 'bayes_opt_lcb_CV',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_srinivas_0.01': {
        'name': 'bayes_opt_lcb_srinivas_0.01',
        'strategy': 'bayes_opt',
        'display_name': 'LCB-Srinivas',
        'bar_group': '0.01',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb-srinivas',
            'methodparams': {
                'explorationfactor': 0.01,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_srinivas_0.1': {
        'name': 'bayes_opt_lcb_srinivas_0.1',
        'strategy': 'bayes_opt',
        'display_name': 'LCB-Srinivas',
        'bar_group': '0.1',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb-srinivas',
            'methodparams': {
                'explorationfactor': 0.1,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_srinivas_0.3': {
        'name': 'bayes_opt_lcb_srinivas_0.3',
        'strategy': 'bayes_opt',
        'display_name': 'LCB-Srinivas',
        'bar_group': '0.3',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb-srinivas',
            'methodparams': {
                'explorationfactor': 0.3,
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_srinivas_CV': {
        'name': 'bayes_opt_lcb_srinivas_CV',
        'strategy': 'bayes_opt',
        'display_name': 'LCB-Srinivas',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb-srinivas',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
}

# 3 Which covariance function lengthscale is optimal for CV?
compare_covariance_functions_CV = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_ei_CV_matern32_0.25': {
        'name': 'bayes_opt_ei_CV_matern32_0.25',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'Matern 3/2 0.25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.25,
        }
    },
    'bayes_opt_ei_CV_matern32_0.5': {
        'name': 'bayes_opt_ei_CV_matern32_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'Matern 3/2 0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.5,
        }
    },
    'bayes_opt_ei_CV': {
        'name': 'bayes_opt_ei_CV',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'Matern 3/2 1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_ei_CV_matern32_1.5': {
        'name': 'bayes_opt_ei_CV_matern32_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'Matern 3/2 1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 1.5,
        }
    },
    'bayes_opt_ei_CV_matern32_2.0': {
        'name': 'bayes_opt_ei_CV_matern32_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'Matern 3/2 2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.0,
        }
    },
    'bayes_opt_ei_CV_matern32_2.5': {
        'name': 'bayes_opt_ei_CV_matern32_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'Matern 3/2 2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.5,
        }
    },
    'bayes_opt_poi_CV_matern32_0.25': {
        'name': 'bayes_opt_poi_CV_matern32_0.25',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'Matern 3/2 0.25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.25,
        }
    },
    'bayes_opt_poi_CV_matern32_0.5': {
        'name': 'bayes_opt_poi_CV_matern32_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'Matern 3/2 0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.5,
        }
    },
    'bayes_opt_poi_CV': {
        'name': 'bayes_opt_poi_CV',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'Matern 3/2 1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_poi_CV_matern32_1.5': {
        'name': 'bayes_opt_poi_CV_matern32_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'Matern 3/2 1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 1.5,
        }
    },
    'bayes_opt_poi_CV_matern32_2.0': {
        'name': 'bayes_opt_poi_CV_matern32_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'Matern 3/2 2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.0,
        }
    },
    'bayes_opt_poi_CV_matern32_2.5': {
        'name': 'bayes_opt_poi_CV_matern32_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'Matern 3/2 2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.5,
        }
    },
    'bayes_opt_lcb_CV_matern32_0.25': {
        'name': 'bayes_opt_lcb_CV_matern32_0.25',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'Matern 3/2 0.25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.25,
        }
    },
    'bayes_opt_lcb_CV_matern32_0.5': {
        'name': 'bayes_opt_lcb_CV_matern32_0.5',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'Matern 3/2 0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 0.5,
        }
    },
    'bayes_opt_lcb_CV': {
        'name': 'bayes_opt_lcb_CV',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'Matern 3/2 1.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
    'bayes_opt_lcb_CV_matern32_1.5': {
        'name': 'bayes_opt_lcb_CV_matern32_1.5',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'Matern 3/2 1.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 1.5,
        }
    },
    'bayes_opt_lcb_CV_matern32_2.0': {
        'name': 'bayes_opt_lcb_CV_matern32_2.0',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'Matern 3/2 2.0',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.0,
        }
    },
    'bayes_opt_lcb_CV_matern32_2.5': {
        'name': 'bayes_opt_lcb_CV_matern32_2.5',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'Matern 3/2 2.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'covariancekernel': 'matern32',
            'covariancelengthscale': 2.5,
        }
    },
}

# from here on it is with Matern 3/2 1.5

# 4A Which skip_duplicate_after is optimal for multi?
compare_skip_threshold_multi = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi_3': {
        'name': 'bayes_opt_multi_3',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '3',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 3
            }
        }
    },
    'bayes_opt_multi_5': {
        'name': 'bayes_opt_multi_5',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
    'bayes_opt_multi_10': {
        'name': 'bayes_opt_multi_10',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '10',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 10
            }
        }
    },
    'bayes_opt_multi_15': {
        'name': 'bayes_opt_multi_15',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '15',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 15
            }
        }
    },
    'bayes_opt_multi_20': {
        'name': 'bayes_opt_multi_20',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '20',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 20
            }
        }
    },
    'bayes_opt_multi_25': {
        'name': 'bayes_opt_multi_25',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 25
            }
        }
    },
    'bayes_opt_multi_30': {
        'name': 'bayes_opt_multi_30',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '30',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    },
}

# 4B Which skip_duplicate_after is optimal for advanced multi?
compare_skip_threshold_multi_advanced = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi-advanced_3': {
        'name': 'bayes_opt_multi-advanced_3',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '3',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 3
            }
        }
    },
    'bayes_opt_multi-advanced_5': {
        'name': 'bayes_opt_multi-advanced_5',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
    'bayes_opt_multi-advanced_10': {
        'name': 'bayes_opt_multi-advanced_10',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '10',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 10
            }
        }
    },
    'bayes_opt_multi-advanced_15': {
        'name': 'bayes_opt_multi-advanced_15',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '15',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 15
            }
        }
    },
    'bayes_opt_multi-advanced_20': {
        'name': 'bayes_opt_multi-advanced_20',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '20',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 20
            }
        }
    },
    'bayes_opt_multi-advanced_25': {
        'name': 'bayes_opt_multi-advanced_25',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 25
            }
        }
    },
    'bayes_opt_multi-advanced_30': {
        'name': 'bayes_opt_multi-advanced_30',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '30',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    }
}

# 4C Which skip_duplicate_after is optimal for advanced precise multi?
compare_skip_threshold_multi_advanced_precise = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi-advanced-precise_3': {
        'name': 'bayes_opt_multi-advanced-precise_3',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '3',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 3
            }
        }
    },
    'bayes_opt_multi-advanced-precise_5': {
        'name': 'bayes_opt_multi-advanced-precise_5',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
    'bayes_opt_multi-advanced-precise_10': {
        'name': 'bayes_opt_multi-advanced-precise_10',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '10',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 10
            }
        }
    },
    'bayes_opt_multi-advanced-precise_15': {
        'name': 'bayes_opt_multi-advanced-precise_15',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '15',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 15
            }
        }
    },
    'bayes_opt_multi-advanced-precise_20': {
        'name': 'bayes_opt_multi-advanced-precise_20',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '20',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 20
            }
        }
    },
    'bayes_opt_multi-advanced-precise_25': {
        'name': 'bayes_opt_multi-advanced-precise_25',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 25
            }
        }
    },
    'bayes_opt_multi-advanced-precise_30': {
        'name': 'bayes_opt_multi-advanced-precise_30',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '30',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 30
            }
        }
    }
}

# compare all CV acquisition functions (Matern 3/2 1.5)
compare_acquisition_functions = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_ei_CV_reference': {
        'name': 'bayes_opt_ei_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
        }
    },
    'bayes_opt_poi_CV_reference': {
        'name': 'bayes_opt_poi_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
        }
    },
    'bayes_opt_lcb_CV_reference': {
        'name': 'bayes_opt_lcb_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
        }
    },
    'bayes_opt_multi-fast': {
        'name': 'bayes_opt_multi-fast',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
        }
    },
    'bayes_opt_multi_5': {
        'name': 'bayes_opt_multi_5',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
    'bayes_opt_multi-advanced_5': {
        'name': 'bayes_opt_multi-advanced_5',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
}

compare_acquisition_functions_multi_advanced = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi_5': {
        'name': 'bayes_opt_multi_5',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
    'bayes_opt_multi-advanced_5_test1': {
        'name': 'bayes_opt_multi-advanced_5_test1',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi 5 0.8',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
    'bayes_opt_multi-advanced_5_test2': {
        'name': 'bayes_opt_multi-advanced_5_test2',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi 5 0.85',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
    'bayes_opt_multi-advanced_5': {
        'name': 'bayes_opt_multi-advanced_5',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi 5 0.9',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
    'bayes_opt_multi-advanced_5_test3': {
        'name': 'bayes_opt_multi-advanced_5_test3',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi 5 0.95',
        'bar_group': 'CV',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'methodparams': {
                'explorationfactor': 'CV',
                'zeta': 1,
                'skip_duplicate_after': 5
            }
        }
    },
}

# 5. What is the optimal order of basic acquisition functions for all multi acquisition functions?
compare_all_multi_optimal_order = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    "bayes_opt_multi-fast_('ei', 'poi', 'lcb')": {
        'name': "bayes_opt_multi-fast_('ei', 'poi', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('ei', 'poi', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['ei', 'poi', 'lcb']
        }
    },
    "bayes_opt_multi-fast_('ei', 'lcb', 'poi')": {
        'name': "bayes_opt_multi-fast_('ei', 'lcb', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('ei', 'lcb', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['ei', 'lcb', 'poi']
        }
    },
    "bayes_opt_multi-fast_('poi', 'ei', 'lcb')": {
        'name': "bayes_opt_multi-fast_('poi', 'ei', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('poi', 'ei', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['poi', 'ei', 'lcb']
        }
    },
    "bayes_opt_multi-fast_('poi', 'lcb', 'ei')": {
        'name': "bayes_opt_multi-fast_('poi', 'lcb', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('poi', 'lcb', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['poi', 'lcb', 'ei']
        }
    },
    "bayes_opt_multi-fast_('lcb', 'ei', 'poi')": {
        'name': "bayes_opt_multi-fast_('lcb', 'ei', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('lcb', 'ei', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['lcb', 'ei', 'poi']
        }
    },
    "bayes_opt_multi-fast_('lcb', 'poi', 'ei')": {
        'name': "bayes_opt_multi-fast_('lcb', 'poi', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('lcb', 'poi', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['lcb', 'poi', 'ei']
        }
    },
    "bayes_opt_multi_('ei', 'poi', 'lcb')": {
        'name': "bayes_opt_multi_('ei', 'poi', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('ei', 'poi', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['ei', 'poi', 'lcb']
        }
    },
    "bayes_opt_multi_('ei', 'lcb', 'poi')": {
        'name': "bayes_opt_multi_('ei', 'lcb', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('ei', 'lcb', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['ei', 'lcb', 'poi']
        }
    },
    "bayes_opt_multi_('poi', 'ei', 'lcb')": {
        'name': "bayes_opt_multi_('poi', 'ei', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('poi', 'ei', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['poi', 'ei', 'lcb']
        }
    },
    "bayes_opt_multi_('poi', 'lcb', 'ei')": {
        'name': "bayes_opt_multi_('poi', 'lcb', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('poi', 'lcb', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['poi', 'lcb', 'ei']
        }
    },
    "bayes_opt_multi_('lcb', 'ei', 'poi')": {
        'name': "bayes_opt_multi_('lcb', 'ei', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('lcb', 'ei', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['lcb', 'ei', 'poi']
        }
    },
    "bayes_opt_multi_('lcb', 'poi', 'ei')": {
        'name': "bayes_opt_multi_('lcb', 'poi', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('lcb', 'poi', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['lcb', 'poi', 'ei']
        }
    },
    "bayes_opt_multi-advanced_('ei', 'poi', 'lcb')": {
        'name': "bayes_opt_multi-advanced_('ei', 'poi', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('ei', 'poi', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['ei', 'poi', 'lcb']
        }
    },
    "bayes_opt_multi-advanced_('ei', 'lcb', 'poi')": {
        'name': "bayes_opt_multi-advanced_('ei', 'lcb', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('ei', 'lcb', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['ei', 'lcb', 'poi']
        }
    },
    "bayes_opt_multi-advanced_('poi', 'ei', 'lcb')": {
        'name': "bayes_opt_multi-advanced_('poi', 'ei', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('poi', 'ei', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['poi', 'ei', 'lcb']
        }
    },
    "bayes_opt_multi-advanced_('poi', 'lcb', 'ei')": {
        'name': "bayes_opt_multi-advanced_('poi', 'lcb', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('poi', 'lcb', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['poi', 'lcb', 'ei']
        }
    },
    "bayes_opt_multi-advanced_('lcb', 'ei', 'poi')": {
        'name': "bayes_opt_multi-advanced_('lcb', 'ei', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('lcb', 'ei', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['lcb', 'ei', 'poi']
        }
    },
    "bayes_opt_multi-advanced_('lcb', 'poi', 'ei')": {
        'name': "bayes_opt_multi-advanced_('lcb', 'poi', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('lcb', 'poi', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['lcb', 'poi', 'ei']
        }
    }
}

# 5A. What is the optimal order of acquisition functions for naive multi?
compare_naive_multi_optimal_order = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    "bayes_opt_multi-fast_('ei', 'poi', 'lcb')": {
        'name': "bayes_opt_multi-fast_('ei', 'poi', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('ei', 'poi', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['ei', 'poi', 'lcb']
        }
    },
    "bayes_opt_multi-fast_('ei', 'lcb', 'poi')": {
        'name': "bayes_opt_multi-fast_('ei', 'lcb', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('ei', 'lcb', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['ei', 'lcb', 'poi']
        }
    },
    "bayes_opt_multi-fast_('poi', 'ei', 'lcb')": {
        'name': "bayes_opt_multi-fast_('poi', 'ei', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('poi', 'ei', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['poi', 'ei', 'lcb']
        }
    },
    "bayes_opt_multi-fast_('poi', 'lcb', 'ei')": {
        'name': "bayes_opt_multi-fast_('poi', 'lcb', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('poi', 'lcb', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['poi', 'lcb', 'ei']
        }
    },
    "bayes_opt_multi-fast_('lcb', 'ei', 'poi')": {
        'name': "bayes_opt_multi-fast_('lcb', 'ei', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('lcb', 'ei', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['lcb', 'ei', 'poi']
        }
    },
    "bayes_opt_multi-fast_('lcb', 'poi', 'ei')": {
        'name': "bayes_opt_multi-fast_('lcb', 'poi', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': "('lcb', 'poi', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'multi_af_names': ['lcb', 'poi', 'ei']
        }
    }
}

# 5B. What is the optimal order of acquisition functions for multi?
compare_multi_optimal_order = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    "bayes_opt_multi_('ei', 'poi', 'lcb')": {
        'name': "bayes_opt_multi_('ei', 'poi', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('ei', 'poi', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['ei', 'poi', 'lcb']
        }
    },
    "bayes_opt_multi_('ei', 'lcb', 'poi')": {
        'name': "bayes_opt_multi_('ei', 'lcb', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('ei', 'lcb', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['ei', 'lcb', 'poi']
        }
    },
    "bayes_opt_multi_('poi', 'ei', 'lcb')": {
        'name': "bayes_opt_multi_('poi', 'ei', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('poi', 'ei', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['poi', 'ei', 'lcb']
        }
    },
    "bayes_opt_multi_('poi', 'lcb', 'ei')": {
        'name': "bayes_opt_multi_('poi', 'lcb', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('poi', 'lcb', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['poi', 'lcb', 'ei']
        }
    },
    "bayes_opt_multi_('lcb', 'ei', 'poi')": {
        'name': "bayes_opt_multi_('lcb', 'ei', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('lcb', 'ei', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['lcb', 'ei', 'poi']
        }
    },
    "bayes_opt_multi_('lcb', 'poi', 'ei')": {
        'name': "bayes_opt_multi_('lcb', 'poi', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': "('lcb', 'poi', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_names': ['lcb', 'poi', 'ei']
        }
    }
}

# 5C. What is the optimal order of acquisition functions for advanced multi?
compare_multi_advanced_optimal_order = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    "bayes_opt_multi-advanced_('ei', 'poi', 'lcb')": {
        'name': "bayes_opt_multi-advanced_('ei', 'poi', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('ei', 'poi', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['ei', 'poi', 'lcb']
        }
    },
    "bayes_opt_multi-advanced_('ei', 'lcb', 'poi')": {
        'name': "bayes_opt_multi-advanced_('ei', 'lcb', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('ei', 'lcb', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['ei', 'lcb', 'poi']
        }
    },
    "bayes_opt_multi-advanced_('poi', 'ei', 'lcb')": {
        'name': "bayes_opt_multi-advanced_('poi', 'ei', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('poi', 'ei', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['poi', 'ei', 'lcb']
        }
    },
    "bayes_opt_multi-advanced_('poi', 'lcb', 'ei')": {
        'name': "bayes_opt_multi-advanced_('poi', 'lcb', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('poi', 'lcb', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['poi', 'lcb', 'ei']
        }
    },
    "bayes_opt_multi-advanced_('lcb', 'ei', 'poi')": {
        'name': "bayes_opt_multi-advanced_('lcb', 'ei', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('lcb', 'ei', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['lcb', 'ei', 'poi']
        }
    },
    "bayes_opt_multi-advanced_('lcb', 'poi', 'ei')": {
        'name': "bayes_opt_multi-advanced_('lcb', 'poi', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': "('lcb', 'poi', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_names': ['lcb', 'poi', 'ei']
        }
    }
}

# 5D. What is the optimal order of acquisition functions for advanced precise multi?
compare_multi_advanced_precise_optimal_order = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    "bayes_opt_multi-advanced-precise_('ei', 'poi', 'lcb')": {
        'name': "bayes_opt_multi-advanced-precise_('ei', 'poi', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': "('ei', 'poi', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_names': ['ei', 'poi', 'lcb']
        }
    },
    "bayes_opt_multi-advanced-precise_('ei', 'lcb', 'poi')": {
        'name': "bayes_opt_multi-advanced-precise_('ei', 'lcb', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': "('ei', 'lcb', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_names': ['ei', 'lcb', 'poi']
        }
    },
    "bayes_opt_multi-advanced-precise_('poi', 'ei', 'lcb')": {
        'name': "bayes_opt_multi-advanced-precise_('poi', 'ei', 'lcb')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': "('poi', 'ei', 'lcb')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_names': ['poi', 'ei', 'lcb']
        }
    },
    "bayes_opt_multi-advanced-precise_('poi', 'lcb', 'ei')": {
        'name': "bayes_opt_multi-advanced-precise_('poi', 'lcb', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': "('poi', 'lcb', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_names': ['poi', 'lcb', 'ei']
        }
    },
    "bayes_opt_multi-advanced-precise_('lcb', 'ei', 'poi')": {
        'name': "bayes_opt_multi-advanced-precise_('lcb', 'ei', 'poi')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': "('lcb', 'ei', 'poi')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_names': ['lcb', 'ei', 'poi']
        }
    },
    "bayes_opt_multi-advanced-precise_('lcb', 'poi', 'ei')": {
        'name': "bayes_opt_multi-advanced-precise_('lcb', 'poi', 'ei')",
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': "('lcb', 'poi', 'ei')",
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_names': ['lcb', 'poi', 'ei']
        }
    }
}

# What is the optimal required improvement factor for advanced multi?
compare_multi_advanced_required_improvement_factor = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi-advanced_0.03': {
        'name': 'bayes_opt_multi-advanced_0.03',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.03',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_afs_required_improvement_factor': 0.03
        }
    },
    'bayes_opt_multi-advanced_0.05': {
        'name': 'bayes_opt_multi-advanced_0.05',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.05',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_afs_required_improvement_factor': 0.05
        }
    },
    'bayes_opt_multi-advanced_0.1': {
        'name': 'bayes_opt_multi-advanced_0.1',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.1',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_afs_required_improvement_factor': 0.1
        }
    },
    'bayes_opt_multi-advanced_0.15': {
        'name': 'bayes_opt_multi-advanced_0.15',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.15',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_afs_required_improvement_factor': 0.15
        }
    },
    'bayes_opt_multi-advanced_0.2': {
        'name': 'bayes_opt_multi-advanced_0.2',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.2',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_afs_required_improvement_factor': 0.2
        }
    },
    'bayes_opt_multi-advanced_0.25': {
        'name': 'bayes_opt_multi-advanced_0.25',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_afs_required_improvement_factor': 0.25
        }
    }
}

# What is the optimal required improvement factor for advanced precise multi?
compare_multi_advanced_precise_required_improvement_factor = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi-advanced-precise_0.03': {
        'name': 'bayes_opt_multi-advanced-precise_0.03',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.03',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_afs_required_improvement_factor': 0.03
        }
    },
    'bayes_opt_multi-advanced-precise_0.05': {
        'name': 'bayes_opt_multi-advanced-precise_0.05',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.05',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_afs_required_improvement_factor': 0.05
        }
    },
    'bayes_opt_multi-advanced-precise_0.1': {
        'name': 'bayes_opt_multi-advanced-precise_0.1',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.1',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_afs_required_improvement_factor': 0.1
        }
    },
    'bayes_opt_multi-advanced-precise_0.15': {
        'name': 'bayes_opt_multi-advanced-precise_0.15',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.15',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_afs_required_improvement_factor': 0.15
        }
    },
    'bayes_opt_multi-advanced-precise_0.2': {
        'name': 'bayes_opt_multi-advanced-precise_0.2',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.2',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_afs_required_improvement_factor': 0.2
        }
    },
    'bayes_opt_multi-advanced-precise_0.25': {
        'name': 'bayes_opt_multi-advanced-precise_0.25',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.25',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_afs_required_improvement_factor': 0.25
        }
    }
}

# What is the optimal discount factor for multi?
compare_multi_discount_factor = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi_df0.5': {
        'name': 'bayes_opt_multi_df0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.5
        }
    },
    'bayes_opt_multi_df0.55': {
        'name': 'bayes_opt_multi_df0.55',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.55',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.55
        }
    },
    'bayes_opt_multi_df0.6': {
        'name': 'bayes_opt_multi_df0.6',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.6',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.6
        }
    },
    'bayes_opt_multi_df0.65': {
        'name': 'bayes_opt_multi_df0.65',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.65',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.65
        }
    },
    'bayes_opt_multi_df0.7': {
        'name': 'bayes_opt_multi_df0.7',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.7',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.7
        }
    },
    'bayes_opt_multi_df0.75': {
        'name': 'bayes_opt_multi_df0.75',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.75',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.75
        }
    },
    'bayes_opt_multi_df0.8': {
        'name': 'bayes_opt_multi_df0.8',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.8',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.8
        }
    },
    'bayes_opt_multi_df0.85': {
        'name': 'bayes_opt_multi_df0.85',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.85',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.85
        }
    },
    'bayes_opt_multi_df0.9': {
        'name': 'bayes_opt_multi_df0.9',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.9',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.9
        }
    },
    'bayes_opt_multi_df0.95': {
        'name': 'bayes_opt_multi_df0.95',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.95',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.95
        }
    },
    'bayes_opt_multi_df0.99': {
        'name': 'bayes_opt_multi_df0.99',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': '0.99',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'multi_af_discount_factor': 0.99
        }
    }
}

# What is the optimal discount factor for advanced multi?
compare_multi_advanced_discount_factor = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi-advanced_df0.5': {
        'name': 'bayes_opt_multi-advanced_df0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.5
        }
    },
    'bayes_opt_multi-advanced_df0.55': {
        'name': 'bayes_opt_multi-advanced_df0.55',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.55',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.55
        }
    },
    'bayes_opt_multi-advanced_df0.6': {
        'name': 'bayes_opt_multi-advanced_df0.6',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.6',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.6
        }
    },
    'bayes_opt_multi-advanced_df0.65': {
        'name': 'bayes_opt_multi-advanced_df0.65',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.65',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.65
        }
    },
    'bayes_opt_multi-advanced_df0.7': {
        'name': 'bayes_opt_multi-advanced_df0.7',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.7',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.7
        }
    },
    'bayes_opt_multi-advanced_df0.75': {
        'name': 'bayes_opt_multi-advanced_df0.75',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.75',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.75
        }
    },
    'bayes_opt_multi-advanced_df0.8': {
        'name': 'bayes_opt_multi-advanced_df0.8',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.8',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.8
        }
    },
    'bayes_opt_multi-advanced_df0.85': {
        'name': 'bayes_opt_multi-advanced_df0.85',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.85',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.85
        }
    },
    'bayes_opt_multi-advanced_df0.9': {
        'name': 'bayes_opt_multi-advanced_df0.9',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.9',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.9
        }
    },
    'bayes_opt_multi-advanced_df0.95': {
        'name': 'bayes_opt_multi-advanced_df0.95',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.95',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.95
        }
    },
    'bayes_opt_multi-advanced_df0.99': {
        'name': 'bayes_opt_multi-advanced_df0.99',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': '0.99',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'multi_af_discount_factor': 0.99
        }
    }
}

# What is the optimal discount factor for advanced precise multi?
compare_multi_advanced_precise_discount_factor = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_multi-advanced-precise_df0.5': {
        'name': 'bayes_opt_multi-advanced-precise_df0.5',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.5',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.5
        }
    },
    'bayes_opt_multi-advanced-precise_df0.55': {
        'name': 'bayes_opt_multi-advanced-precise_df0.55',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.55',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.55
        }
    },
    'bayes_opt_multi-advanced-precise_df0.6': {
        'name': 'bayes_opt_multi-advanced-precise_df0.6',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.6',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.6
        }
    },
    'bayes_opt_multi-advanced-precise_df0.65': {
        'name': 'bayes_opt_multi-advanced-precise_df0.65',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.65',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.65
        }
    },
    'bayes_opt_multi-advanced-precise_df0.7': {
        'name': 'bayes_opt_multi-advanced-precise_df0.7',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.7',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.7
        }
    },
    'bayes_opt_multi-advanced-precise_df0.75': {
        'name': 'bayes_opt_multi-advanced-precise_df0.75',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.75',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.75
        }
    },
    'bayes_opt_multi-advanced-precise_df0.8': {
        'name': 'bayes_opt_multi-advanced-precise_df0.8',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.8',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.8
        }
    },
    'bayes_opt_multi-advanced-precise_df0.85': {
        'name': 'bayes_opt_multi-advanced-precise_df0.85',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.85',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.85
        }
    },
    'bayes_opt_multi-advanced-precise_df0.9': {
        'name': 'bayes_opt_multi-advanced-precise_df0.9',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.9',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.9
        }
    },
    'bayes_opt_multi-advanced-precise_df0.95': {
        'name': 'bayes_opt_multi-advanced-precise_df0.95',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.95',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.95
        }
    },
    'bayes_opt_multi-advanced-precise_df0.99': {
        'name': 'bayes_opt_multi-advanced-precise_df0.99',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Precise Multi',
        'bar_group': '0.99',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced-precise',
            'multi_af_discount_factor': 0.99
        }
    }
}

# what is the optimal initial sampling for all acquisition functions?
compare_initial_sampling = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_ei_ISrandom': {
        'name': 'bayes_opt_ei_ISrandom',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'random',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'samplingmethod': 'random'
        }
    },
    'bayes_opt_ei_ISNone': {
        'name': 'bayes_opt_ei_ISNone',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'centered',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'samplingcriterion': None
        }
    },
    'bayes_opt_ei_IScorrelation': {
        'name': 'bayes_opt_ei_IScorrelation',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'correlation',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'samplingcriterion': 'correlation'
        }
    },
    'bayes_opt_ei_ISmaximin': {
        'name': 'bayes_opt_ei_ISmaximin',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'maximin',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'samplingcriterion': 'maximin'
        }
    },
    'bayes_opt_ei_ISratio': {
        'name': 'bayes_opt_ei_ISratio',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'ratio',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'samplingcriterion': 'ratio'
        }
    },
    'bayes_opt_poi_ISrandom': {
        'name': 'bayes_opt_poi_ISrandom',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'random',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'samplingmethod': 'random'
        }
    },
    'bayes_opt_poi_ISNone': {
        'name': 'bayes_opt_poi_ISNone',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'centered',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'samplingcriterion': None
        }
    },
    'bayes_opt_poi_IScorrelation': {
        'name': 'bayes_opt_poi_IScorrelation',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'correlation',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'samplingcriterion': 'correlation'
        }
    },
    'bayes_opt_poi_ISmaximin': {
        'name': 'bayes_opt_poi_ISmaximin',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'maximin',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'samplingcriterion': 'maximin'
        }
    },
    'bayes_opt_poi_ISratio': {
        'name': 'bayes_opt_poi_ISratio',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'ratio',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'samplingcriterion': 'ratio'
        }
    },
    'bayes_opt_lcb_ISrandom': {
        'name': 'bayes_opt_lcb_ISrandom',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'random',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'samplingmethod': 'random'
        }
    },
    'bayes_opt_lcb_ISNone': {
        'name': 'bayes_opt_lcb_ISNone',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'centered',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'samplingcriterion': None
        }
    },
    'bayes_opt_lcb_IScorrelation': {
        'name': 'bayes_opt_lcb_IScorrelation',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'correlation',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'samplingcriterion': 'correlation'
        }
    },
    'bayes_opt_lcb_ISmaximin': {
        'name': 'bayes_opt_lcb_ISmaximin',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'maximin',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'samplingcriterion': 'maximin'
        }
    },
    'bayes_opt_lcb_ISratio': {
        'name': 'bayes_opt_lcb_ISratio',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'ratio',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'samplingcriterion': 'ratio'
        }
    },
    'bayes_opt_multi-fast_ISrandom': {
        'name': 'bayes_opt_multi-fast_ISrandom',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'random',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'samplingmethod': 'random'
        }
    },
    'bayes_opt_multi-fast_ISNone': {
        'name': 'bayes_opt_multi-fast_ISNone',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'centered',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'samplingcriterion': None
        }
    },
    'bayes_opt_multi-fast_IScorrelation': {
        'name': 'bayes_opt_multi-fast_IScorrelation',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'correlation',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'samplingcriterion': 'correlation'
        }
    },
    'bayes_opt_multi-fast_ISmaximin': {
        'name': 'bayes_opt_multi-fast_ISmaximin',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'maximin',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'samplingcriterion': 'maximin'
        }
    },
    'bayes_opt_multi-fast_ISratio': {
        'name': 'bayes_opt_multi-fast_ISratio',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'ratio',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'samplingcriterion': 'ratio'
        }
    },
    'bayes_opt_multi_ISrandom': {
        'name': 'bayes_opt_multi_ISrandom',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'random',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'samplingmethod': 'random'
        }
    },
    'bayes_opt_multi_ISNone': {
        'name': 'bayes_opt_multi_ISNone',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'centered',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'samplingcriterion': None
        }
    },
    'bayes_opt_multi_IScorrelation': {
        'name': 'bayes_opt_multi_IScorrelation',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'correlation',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'samplingcriterion': 'correlation'
        }
    },
    'bayes_opt_multi_ISmaximin': {
        'name': 'bayes_opt_multi_ISmaximin',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'maximin',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'samplingcriterion': 'maximin'
        }
    },
    'bayes_opt_multi_ISratio': {
        'name': 'bayes_opt_multi_ISratio',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'ratio',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'samplingcriterion': 'ratio'
        }
    },
    'bayes_opt_multi-advanced_ISrandom': {
        'name': 'bayes_opt_multi-advanced_ISrandom',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'random',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'samplingmethod': 'random'
        }
    },
    'bayes_opt_multi-advanced_ISNone': {
        'name': 'bayes_opt_multi-advanced_ISNone',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'centered',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'samplingcriterion': None
        }
    },
    'bayes_opt_multi-advanced_IScorrelation': {
        'name': 'bayes_opt_multi-advanced_IScorrelation',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'correlation',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'samplingcriterion': 'correlation'
        }
    },
    'bayes_opt_multi-advanced_ISmaximin': {
        'name': 'bayes_opt_multi-advanced_ISmaximin',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'maximin',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'samplingcriterion': 'maximin'
        }
    },
    'bayes_opt_multi-advanced_ISratio': {
        'name': 'bayes_opt_multi-advanced_ISratio',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'ratio',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'samplingcriterion': 'ratio'
        }
    }
}

# What is better: pruned or not?
compare_acquisition_functions_pruned = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_ei_CV_notpruned': {
        'name': 'bayes_opt_ei_CV_notpruned',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'default',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
            'pruneparameterspace': False,
        }
    },
    'bayes_opt_poi_CV_notpruned': {
        'name': 'bayes_opt_poi_CV_notpruned',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'default',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
            'pruneparameterspace': False,
        }
    },
    'bayes_opt_lcb_CV_notpruned': {
        'name': 'bayes_opt_lcb_CV_notpruned',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'default',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
            'pruneparameterspace': False,
        }
    },
    'bayes_opt_multi-fast_notpruned': {
        'name': 'bayes_opt_multi-fast_notpruned',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'default',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
            'pruneparameterspace': False,
        }
    },
    'bayes_opt_multi_notpruned': {
        'name': 'bayes_opt_multi_notpruned',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'default',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
            'pruneparameterspace': False,
        }
    },
    'bayes_opt_multi-advanced_notpruned': {
        'name': 'bayes_opt_multi-advanced_notpruned',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'default',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 7,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
            'pruneparameterspace': False,
        }
    },
    'bayes_opt_ei_CV_reference': {
        'name': 'bayes_opt_ei_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'pruned',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
        }
    },
    'bayes_opt_poi_CV_reference': {
        'name': 'bayes_opt_poi_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'pruned',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
        }
    },
    'bayes_opt_lcb_CV_reference': {
        'name': 'bayes_opt_lcb_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'pruned',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
        }
    },
    'bayes_opt_multi-fast_reference': {
        'name': 'bayes_opt_multi-fast_reference',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'pruned',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
        }
    },
    'bayes_opt_multi_reference': {
        'name': 'bayes_opt_multi_reference',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'pruned',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
        }
    },
    'bayes_opt_multi-advanced_reference': {
        'name': 'bayes_opt_multi-advanced_reference',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'pruned',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
        }
    },
}

# What is the optimal acquisition function?
compare_acquisition_functions = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_ei_CV_reference': {
        'name': 'bayes_opt_ei_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'basic',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
        }
    },
    'bayes_opt_poi_CV_reference': {
        'name': 'bayes_opt_poi_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'PoI',
        'bar_group': 'basic',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'poi',
        }
    },
    'bayes_opt_lcb_CV_reference': {
        'name': 'bayes_opt_lcb_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'LCB',
        'bar_group': 'basic',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'lcb',
        }
    },
    'bayes_opt_multi-fast_reference': {
        'name': 'bayes_opt_multi-fast_reference',
        'strategy': 'bayes_opt',
        'display_name': 'Naive Multi',
        'bar_group': 'multi',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-fast',
        }
    },
    'bayes_opt_multi_reference': {
        'name': 'bayes_opt_multi_reference',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'multi',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
        }
    },
    'bayes_opt_multi-advanced_reference': {
        'name': 'bayes_opt_multi-advanced_reference',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'multi',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
        }
    },
}

# What is the optimal selected acquisition function?
compare_selected_acquisition_functions = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random Sample',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'bayes_opt_ei_CV_reference': {
        'name': 'bayes_opt_ei_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'basic',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'ei',
        }
    },
    'bayes_opt_multi_reference': {
        'name': 'bayes_opt_multi_reference',
        'strategy': 'bayes_opt',
        'display_name': 'Multi',
        'bar_group': 'multi',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi',
        }
    },
    'bayes_opt_multi-advanced_reference': {
        'name': 'bayes_opt_multi-advanced_reference',
        'strategy': 'bayes_opt',
        'display_name': 'Advanced Multi',
        'bar_group': 'multi',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
        }
    },
}
