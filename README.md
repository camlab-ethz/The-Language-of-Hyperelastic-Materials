# The Language of Hyperelastic Materials 

## Abstract: 

The automated discovery of constitutive laws forms an emerging research area, that focuses on automatically obtaining symbolic expressions describing the constitutive behavior of solid materials from experimental data. Existing symbolic/sparse regression methods rely on the availability of libraries of material models, which are typically hand-designed by a human expert using known models as reference, or deploy generative algorithms with exponential complexity which are only practicable for very simple expressions. In this paper, we propose a novel approach to constitutive law discovery relying on formal grammars as an automated and systematic tool to generate constitutive law expressions. Compliance with physics constraints is partly enforced a priori and partly empirically checked a posteriori. We deploy the approach for two tasks: i) Automatically generating a library of valid constitutive laws for hyperelastic isotropic materials; ii) Performing data-driven discovery of hyperelastic material models from displacement data affected by different noise levels. For the task of automatic library generation, we demonstrate the flexibility and efficiency of the proposed methodology in avoiding hand-crafted features and human intervention. For the data-driven discovery task, we demonstrate the accuracy, robustness and significant generalizability of the proposed methodology.

This is the code accompanying the manuscript "The Language of Hyperelastic Materials" soon to appear in "Computer Methods in Applied Mechanics and Engineering". 

Training the model can be performed using the interactive notebook ```train_model.ipynb``` and the discovery process can be performed using the interactive python notebook  ```hyperelastic_discovery.ipynb```. The results can be plotted using the post_process notebooks ```post_process_results*.ipynb```. The ```post_process_*``` files containing ```Flaschel``` to their name correspond to the unsupervised and the other to the supervised discovery. In the folder ```Data``` the Finite Element Method simulation data accompanying the manuscript can be found. In the folder ```results``` one can find a trained ```pytorch``` model that can be used to re-produce the discovery results presented in the paper. Moreover, we include the notebook ```hyperelasticity_plate_grammarmodels.ipynb``` that contains a minimal Fenics implementation for solving the plate with a hole problem described in the paper with using different constitutive law models. The script ```parallel_discovery.py```
can be used to perform the discovery process in paralled using multithreading. The code ```constitutive_relation_sampling.ipynb``` is an interactive notebook for sampling different constitutive laws from the grammar defined in ```constitutive_data.py```

How to cite:

    @article{kissas2024language,
    title={The language of hyperelastic materials},
    author={Kissas, Georgios and Mishra, Siddhartha and Chatzi, Eleni and De Lorenzis, Laura},
    journal={Computer Methods in Applied Mechanics and Engineering},
    volume={428},
    pages={117053},
    year={2024},
    publisher={Elsevier}
    }
