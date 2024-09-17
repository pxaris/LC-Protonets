# LC-Protonets: Multi-label Few-shot learning

PyTorch implementation of **LC-Protonets**, a method designed for multi-label few-shot learning. This implementation is applied specifically to the task of music tagging, with an emphasis on world music.


## Reference

[**LC-Protonets: Multi-label Few-shot learning for world music audio tagging**](https://arxiv.org/).  
- Charilaos Papaioannou, Emmanouil Benetos, and Alexandros Potamianos


## Requirements

* Python 3.10 or later
* To set up the environment and install the necessary packages, run:
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## Data preparation

Follow the steps described in the sections [Datasets](https://github.com/pxaris/ccml?tab=readme-ov-file#datasets) and [Preprocessing](https://github.com/pxaris/ccml?tab=readme-ov-file#preprocessing) from the [ccml](https://github.com/pxaris/ccml) repository for data preparation. This includes downloading the required datasets, processing audio features, and organizing the data in a format compatible with LC-Protonets.


## Models

### Backbone model pre-training

The VGG-ish model serves as the feature extractor for the LC-Protonets framework. Pre-training this model through supervised learning (on each audio dataset) ensures that the network has learned good representations of audio data before being fine-tuned for few-shot tasks. For details on this pre-training process, refer to the [Training](https://github.com/pxaris/ccml?tab=readme-ov-file#training) section of the [ccml](https://github.com/pxaris/ccml) repository.


### Download trained models

All trained models, including pre-trained backbones, can be downloaded from: [LC_Protonets__saved_models.zip](https://drive.google.com/file/d/1knRTbp_5U6K7ezFZvtE6MqLdldua1tYh/view?usp=drive_link) (771 MB).

To use these models, extract the contents of the zip file to the `saved_models/` directory in your project. These models can be loaded during training or evaluation by specifying the appropriate file paths.


## Training

To train the models, use the `few_shot_train.py` script. Below is a detailed explanation of the command-line arguments that can be used to configure the training process:

### Arguments

- `--dataset`:  
  The name of the dataset to use for training.  
  **Options**: `magnatagatune`, `fma`, `makam`, `lyra`, `hindustani`, `carnatic`  
  **Default**: `lyra`

- `--data_dir`:  
  The directory where the **mel-spectrograms** and **split** directories are expected to be found. If not specified, the script will look for the `lyra` dataset in the default data directory.  
  **Default**: `os.path.join(DATA_DIR, 'lyra')`

- `--method`:  
  The method to be used for multi-label few-shot learning.  
  **Options**:
  - `baseline`: Multi-label Prototypical Networks (ML-PNs)
  - `OvR`: One-vs.-Rest
  - `LCP`: Label-Combination Prototypical Networks (LC-Protonets)  
  **Default**: `LCP`

- `--backbone_model`:  
  The backbone architecture to be used for feature extraction.  
  **Options**: `vgg_ish` (VGG-ish model)  
  **Default**: `vgg_ish`

- `--dist`:  
  The distance metric to use for prototype-based classification.  
  **Options**: 
  - `l2`: Euclidean distance
  - `cos`: Cosine distance  
  **Default**: `cos`

- `--source`:  
  Optionally, define a dataset to load a pre-trained model from a specific directory.  
  **Options**: All datasets listed under `DATASETS` or `None`  
  **Default**: `None`

- `--freeze`:  
  Whether to freeze the weights of the backbone model except for the final embedding layer. This allows the model to retain previously learned features while fine-tuning for the specific task.  
  **Default**: `False`  
  **Usage**: You can set this flag to freeze the backbone model by using `--freeze` without providing a value, or explicitly set it to `True` or `False`.

- `--run_idx`:  
  Define a run index to vary the random seed used for different runs. This is useful for performing multiple training runs with different seeds.  
  **Options**: `1`, `2`, `3`, `4`, `5`  
  **Default**: `1`

- `--device`:  
  Specify the device to use for training. Set this to `cpu` if no GPU is available, or choose a specific GPU by specifying the CUDA index (e.g., `cuda:0`, `cuda:1`).  
  **Default**: `cuda:0`


### Example commands

- **training from scratch** (random weights initialization) on MagnaTagATune dataset, using "LC-Protonets" method:
```bash
python few_shot_train.py --dataset "magnatagatune" --data_dir "/__path_to__/magnatagatune" --method "LCP" --dist "cos" --run_idx "1" --device 'cuda:0'
```

- **training with a pre-trained backbone and full fine-tuning** on Lyra dataset, using "One-vs.-Rest" method:
```bash
python few_shot_train.py --dataset "lyra" --data_dir "/__path_to__/lyra" --method "OvR" --dist "cos" --source "lyra" --freeze "False" --run_idx "1" --device 'cuda:0'
```

- **training with a pre-trained backbone and fine-tuning of the last layer** on FMA-medium dataset, using "ML-PNs" method:
```bash
python few_shot_train.py --dataset "fma" --data_dir "/__path_to__/fma" --method "baseline" --dist "cos" --source "fma" --freeze "True" --run_idx "1" --device 'cuda:0'
```

### Naming Conventions of the Saved Models

The models will be saved in the `saved_models/{dataset}` directory using the following naming conventions:

- `{method}_{dist}.pth`: For training from scratch.
- `{method}_from_{dataset}_{dist}.pth`: For training with a pre-trained backbone and full fine-tuning.
- `{method}_from_{dataset}_f_{dist}.pth`: For training with a pre-trained backbone and fine-tuning of the last layer.


## Evaluation

To evaluate the models, use the `few_shot_evaluate.py` script. The command-line arguments are the following:


### Arguments

Same as in training:
- `--dataset`
- `--data_dir`
- `--method`
- `--dist`
- `--device`
- `--run_idx`

- `--model`:  
  The trained model to be used for evaluation.  
  **Default**: `baseline`

- `--N`:  
  The number of tags/labels to include in the evaluation (N-way).
  **Default**: `5`

- `--K`:  
  The number of support items (examples) per label (K-shot).  
  **Default**: `3`

- `--type`:  
  Specifies whether to evaluate on "Base" classes (seen during training), "Novel" classes (not seen during training), or both "Base & Novel".  
  **Options**: 
  - `base`: Use "Base" classes (classes seen during training).
  - `novel`: Use "Novel" classes (classes not seen during training).
  - `both`: Use both "Base & Novel" classes.  
  **Default**: `novel`

- `--source`:  
  The directory from which to load the pre-trained model. For example, you can load a model from the `magnatagatune` dataset or a pre-trained model directory like `pretrained/makam`.  
  **Default**: `magnatagatune`


### Example commands

We use "magnatagatune" dataset for the following evaluation examples.

- evaluate on a `5-way 3-shot` task with `novel` classes, a **from scratch** model that was trained with "LC-Protonets" method:
```bash
python few_shot_evaluate.py --dataset "magnatagatune" --data_dir "/__path_to__/magnatagatune" --method "LCP" --dist "cos" --model "LCP" --N "5" --K "3" --type "novel" --source "magnatagatune" --run_idx "1" --device 'cuda:0' 
```

- evaluate on a `15-way 3-shot` task with `novel` classes, a **full fine-tuning** model that was trained with "ML-PNs" method:
```bash
python few_shot_evaluate.py --dataset "magnatagatune" --data_dir "/__path_to__/magnatagatune" --method "baseline" --dist "cos" --model "baseline_from_magnatagatune" --N "15" --K "3" --type "novel" --source "magnatagatune" --run_idx "1" --device 'cuda:0'
```

- evaluate on a `30-way 3-shot` task with `both` classes, a **fine-tuning of the last layer** model that was trained with "One-vs.-Rest" method:
```bash
python few_shot_evaluate.py --dataset "magnatagatune" --data_dir "/__path_to__/magnatagatune" --method "OvR" --dist "cos" --model "OvR_from_magnatagatune" --N "30" --K "3" --type "both" --source "magnatagatune" --run_idx "1" --device 'cuda:0'
```

- evaluate on a `60-way 3-shot` task with `both` classes, a **pre-trained without any fine-tuning** model using the "LC-Protonets" method on top of the pre-trained `vgg_ish` backbone:
```bash
python few_shot_evaluate.py --dataset "magnatagatune" --data_dir "/__path_to__/magnatagatune" --method "LCP" --dist "cos" --model "vgg_ish" --N "60" --K "3" --type "both" --source "pretrained/magnatagatune" --run_idx "1" --device 'cuda:0'
```

### Naming Conventions of the Evaluation Results

The evaluation results are saved in the `evaluation/{dataset}` directory using the following naming conventions:

- `{N}_way_{type}_{method}_{dist}.pth`: For evaluating a model trained "from scratch".
- `{N}_way_{type}_{method}_from_{dataset}_{dist}.pth`: For evaluating a model with "full fine-tuning".
- `{N}_way_{type}_{method}_from_{dataset}_f_{dist}.pth`: For evaluating a model with "fine-tuning of the last layer".
- `{N}_way_{type}_{method}_pretrained_vgg_ish_{dist}.pth`: For evaluating a "pre-trained without any fine-tuning" model.

Each evaluation file contains:
- Macro-F1 and Micro-F1 scores
- A per-tag classification report for the model
- Process insights:
    - The number of prototypes (an aspect of interest for the LC-Protonets method)
    - The number of unique items in the support set and the query set
    - The mean ground truth and predicted labels per item
    - Total execution time (particularly useful for evaluating the scalability of the LC-Protonets method)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

