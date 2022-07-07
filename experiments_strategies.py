# Swap these dictionaries for the one set to self.strategies in experiments.py

# What is the optimal search method?
# Basinhopping is not used because it performs very poorly and does not work on the adding kernel
compare_search_method = {
    'random_sample': {
        'name': 'random_sample',
        'strategy': 'random_sample',
        'display_name': 'Random',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 100,
        'options': {
            'fraction': 0.05
        }
    },
    'simulated_annealing': {
        'name': 'simulated_annealing',
        'strategy': 'simulated_annealing',
        'display_name': 'SA',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220
        }
    },
    'basinhopping': {
        'name': 'basinhopping',
        'strategy': 'basinhopping',
        'display_name': 'BH',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'maxiter': 150,
        }
    },
    'mls': {
        'name': 'mls',
        'strategy': 'mls',
        'display_name': 'MLS',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
        }
    },
    'genetic_algorithm': {
        'name': 'genetic_algorithm',
        'strategy': 'genetic_algorithm',
        'display_name': 'GA',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220
        }
    },
    'bayes_opt_ei_CV_reference': {
        'name': 'bayes_opt_ei_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'bo',
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
        'bar_group': 'bo',
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
        'bar_group': 'bo',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
        }
    },
}

# What is the optimal long-term search method?
compare_search_method_extended_old = {
    'extended_random_sample': {
        'name':
        'extended_random_sample',
        'strategy':
        'random_sample',
        'display_name':
        'Random Sample',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        100,
        'options': {
            'fraction': 0.1
        }
    },
    'extended_pso': {
        'name':
        'extended_pso',
        'strategy':
        'pso',
        'display_name':
        'PSO',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        35,
        'options': {
            'max_fevals': 1020
        }
    },
    'extended_simulated_annealing': {
        'name':
        'extended_simulated_annealing',
        'strategy':
        'simulated_annealing',
        'display_name':
        'SA',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        35,
        'options': {
            'max_fevals': 1020
        }
    },
    'extended_firefly': {
        'name':
        'extended_firefly',
        'strategy':
        'firefly_algorithm',
        'display_name':
        'FA',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        35,
        'options': {
            'max_fevals': 1020
        }
    },
    'extended_genetic_algorithm': {
        'name':
        'extended_genetic_algorithm',
        'strategy':
        'genetic_algorithm',
        'display_name':
        'GA',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        35,
        'options': {
            'max_fevals': 1020
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
        'bar_group': 'bo',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
        }
    },
}

# What is the optimal long-term search method?
# Basinhopping is not used because it performs very poorly and does not work on the adding kernel
compare_search_method_extended = {
    'extended_random_sample_rem_unique': {
        'name':
        'extended_random_sample_rem_unique',
        'strategy':
        'random_sample',
        'display_name':
        'Random',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        100,
        'options': {
            'fraction': 0.5
        }
    },
    'extended_simulated_annealing_rem_unique': {
        'name':
        'extended_simulated_annealing_rem_unique',
        'strategy':
        'simulated_annealing',
        'display_name':
        'SA',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        35,
        'options': {
            'max_fevals': 1020
        }
    },
    'extended_mls_rem_unique': {
        'name':
        'extended_mls_rem_unique',
        'strategy':
        'mls',
        'display_name':
        'MLS',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        35,
        'options': {
            'max_fevals': 1020,
        }
    },
    'extended_basinhopping_rem_unique': {
        'name':
        'extended_basinhopping_rem_unique',
        'strategy':
        'basinhopping',
        'display_name':
        'BH',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        35,
        'options': {
            'max_fevals': 1020,
            'maxiter': 400,
        }
    },
    'extended_genetic_algorithm_rem_unique': {
        'name':
        'extended_genetic_algorithm_rem_unique',
        'strategy':
        'genetic_algorithm',
        'display_name':
        'GA',
        'bar_group':
        'reference',
        'nums_of_evaluations': [
            20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
            620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
        ],
        'repeats':
        35,
        'options': {
            'max_fevals': 1020,
            'maxiter': 200,
            'popsize': 80,
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
        'bar_group': 'bo',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
        }
    },
}

# What is the optimal BO implementation?
compare_alternative_frameworks = {
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
    'bayes_opt_alt_bayesopt': {
        'name': 'bayes_opt_alt_bayesopt',
        'strategy': 'bayes_opt_alt_bayesopt',
        'display_name': 'Alt BayesOpt',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
        }
    },
    'bayes_opt_alt_scikitopt': {
        'name': 'bayes_opt_alt_scikitopt',
        'strategy': 'bayes_opt_alt_scikitopt',
        'display_name': 'Alt ScikitOpt',
        'bar_group': 'reference',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
        }
    },
    'bayes_opt_ei_CV_reference': {
        'name': 'bayes_opt_ei_CV_reference',
        'strategy': 'bayes_opt',
        'display_name': 'EI',
        'bar_group': 'bo',
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
        'bar_group': 'bo',
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
        'bar_group': 'bo',
        'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        'repeats': 35,
        'options': {
            'max_fevals': 220,
            'method': 'multi-advanced',
        }
    },
}
