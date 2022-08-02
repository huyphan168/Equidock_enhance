# Source code for EquiDock: Independent SE(3)-Equivariant Models for End-to-End Rigid Protein Docking (ICLR 2022)

![EquiDock banner and concept](https://github.com/octavian-ganea/equidock_public/blob/main/equidock.png)


Please cite
```angular2html
@article{ganea2021independent,
  title={Independent SE (3)-Equivariant Models for End-to-End Rigid Protein Docking},
  author={Ganea, Octavian-Eugen and Huang, Xinyuan and Bunne, Charlotte and Bian, Yatao and Barzilay, Regina and Jaakkola, Tommi and Krause, Andreas},
  journal={arXiv preprint arXiv:2111.07786},
  year={2021}
}
```


## Dependencies
Current code works on Linux/Mac OSx only, you need to modify file paths to work on Windows.
```
python==3.9.10
numpy==1.22.1
cuda==10.1
torch==1.10.2
dgl==0.7.0
biopandas==0.2.8
ot==0.7.0
rdkit==2021.09.4
dgllife==0.2.8
joblib==1.1.0
```



## DB5.5 data

The raw DB5.5 dataset was already placed in the `data` directory from the original source:
```
https://zlab.umassmed.edu/benchmark/ or https://github.com/drorlab/DIPS
```
The raw pdb files of DB5.5 dataset are in the directory `./data/benchmark5.5/structures`

Then preprocess the raw data as follows to prepare data for rigid body docking:
```
# prepare data for rigid body docking
python preprocess_raw_data.py -n_jobs 40 -data db5 -graph_nodes residues -graph_cutoff 30 -graph_max_neighbor 10 -graph_residue_loc_is_alphaC -pocket_cutoff 8
```

By default, `preprocess_raw_data.py` uses 10 neighbor for each node when constructing
the graph and uses only residues (coordinates being those of the alpha carbons). After running `preprocess_raw_data.py` you will get following 
ready-for-training data directory:

```
./cache/db5_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0/cv_0/
```
with files
```
$ ls cache/db5_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0/cv_0/
label_test.pkl			label_val.pkl			ligand_graph_train.bin		receptor_graph_test.bin		receptor_graph_val.bin
label_train.pkl			ligand_graph_test.bin		ligand_graph_val.bin		receptor_graph_train.bin
```


## DIPS data
The raw data `DIPS/data/DIPS/interim/pairs-pruned/` can be downloaded from https://www.dropbox.com/s/sqknqofy58nlosh/DIPS.zip?dl=0 
Please unzip the file and put it in the `data` directory.
Finally, preprocess the raw data as follow to prepare data for rigid body docking:
The original preprocess script is extremely computational intensive, especially the memory (300gb of RAM is a must). However by integrating
some redundant process, we save it to only 150gb of RAM and it taks 2 hours to complete the whole process on 48-core CPU.
```
# prepare data for rigid body docking
python preprocess_raw_data.py -n_jobs 48 -data dips -graph_nodes residues -graph_cutoff 30 -graph_max_neighbor 10 -graph_residue_loc_is_alphaC -pocket_cutoff 8 -data_fraction 1.0
```

You should now obtain the following cache data directory:
```angular2html
$ ls cache/dips_residues_maxneighbor_10_cutoff_30.0_pocketCut_8.0/cv_0/
label_test.pkl		     ligand_graph_val.bin		  receptor_graph_frac_1.0_train.bin
label_val.pkl		     ligand_graph_frac_1.0_train.bin  receptor_graph_test.bin
label_frac_1.0_train.pkl   ligand_graph_test.bin	      receptor_graph_val.bin
```



## Training
On multiple GPU (notes that we can not use hyper-search since it will create a asynchronism of parameters between DDP processes (we're working on it):
```angular2html
python -m torch.distributed.launch --nproc_per_node 4 src/train.py  -device 0,1,2,3 -bs 16
```
or just specify your own params if you don't want to do hyperparam search. This will create checkpoints, tensorboard logs (you can visualize with tensorboard) and will store all stdout/stderr in a log file. This will train a model on DIPS first and, then, fine-tune it on DB5. Use `-toy` to train on DB5 only.

## Data splits
In our paper, we used the train/validation/test splits given by the files
```angular2html
DIPS: DIPS/data/DIPS/interim/pairs-pruned/pairs-postprocessed-*.txt
DB5: data/benchmark5.5/cv/cv_0/*.txt
```

## Inference

See `inference_rigid.py`.

## Pretrained models
Our paper pretrained models are available in folder `checkpts/`. By loading those (as in `inference_rigid.py`), you can also see 
which hyperparameters were used in those models (or directly from their names).

## Test and reproduce paper's numbers
Test sets used in our paper are given in `test_sets_pdb/`. Ground truth (bound) structures are in `test_sets_pdb/dips_test_random_transformed/complexes/`, 
while unbound structures (i.e., randomly rotated and translated ligands and receptors) are in `test_sets_pdb/dips_test_random_transformed/random_transformed/` 
and you should precisely use those for your predictions (or at least the ligands, while using the ground truth receptors like we do in `inference_rigid.py`). 
This test set was originally generated as a randomly sampled family-based subset of complexes in `./DIPS/data/DIPS/interim/pairs-pruned/pairs-postprocessed-test.txt`
using the file `src/test_all_methods/testset_random_transf.py`.


Run `python -m src.inference_rigid` to produce EquiDock's outputs for all test files. This will create a new directory of PDB output files in `test_sets_pdb/`. 

Get RMSD numbers from our papers using `python -m src.test_all_methods.eval_pdb_outputset`. You can use this script to evaluate all other baselines. Baselines' output PDB files are also provided in  `test_sets_pdb/`

### Note on steric clashes
Some clashes are possible in our model and we are working on mitigating this issue. Our current solution is a postprocessing clash removal step in `inference_rigid.py#L19`. Output files for DIPS are in `test_sets_pdb/dips_equidock_no_clashes_results/` and for DB5 in `test_sets_pdb/db5_equidock_no_clashes_results/`. 




