# HMM + DualSimplex: Probabilistic Clustering of MHC Class II Peptide Mixtures

Unsupervised deconvolution of mixed MHC class II peptide repertoires into binding motifs,
combining a hierarchical ensemble of anchor-aware Hidden Markov Models with DualSimplex
matrix factorization.

This repository contains the full pipeline accompanying the paper *"Clustering peptide
sequences using Hidden Markov Models and matrix factorization methods"* (Klimova, Kleverov,
Shalyto, Artyomov). It covers data generation and preprocessing, hierarchical HMM training,
DualSimplex clustering, and comparison with GibbsCluster 2.0.

---

## Table of contents

- [Background](#background)
- [Method overview](#method-overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Pipeline](#pipeline)
  - [`000_make_iedb_dataset.py` - IEDB preprocessing](#000_make_iedb_datasetpy--iedb-preprocessing)
  - [`001_generate_simulated_motifs.py` - synthetic data generation](#001_generate_simulated_motifspy--synthetic-data-generation)
  - [`002_make_per_allele_per_length_split.py` - per-length / k-fold split](#002_make_per_allele_per_length_splitpy--per-length--k-fold-split)
  - [`simulated_main.ipynb` / `main_IEDB.ipynb` - hierarchical HMM training](#simulated_mainipynb--main_iedbipynb--hierarchical-hmm-training)
  - [`dual_simplex.R` - DualSimplex clustering and benchmark](#dual_simplexr--dualsimplex-clustering-and-benchmark)
- [How DualSimplex is applied](#how-dualsimplex-is-applied)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Modules](#modules)
- [Configuration](#configuration)
- [References & acknowledgements](#references--acknowledgements)

---

## Background

MHC class II molecules present peptides to CD4⁺ T cells. Their binding groove is open at
both ends, so ligands vary in length (here 12-18 residues) and only a central **9-mer
core** determines binding, with the main preferences at anchor positions P1, P4, P6 and P9.
The position of the core inside a peptide is unknown.

In mass-spectrometry immunopeptidomics, eluted ligands are *mixtures*: peptides bound by
several co-expressed HLA molecules, with no per-peptide allele label. **Motif
deconvolution** splits such a mixture back into its binding motifs and estimates how
strongly each peptide belongs to each of them. Because several motifs can be compatible
with the same peptide (and motifs of different alleles overlap), the task is framed as
**soft (probabilistic) clustering**.

## Method overview

The method has two stages.

**Stage 1 - hierarchical HMM ensemble (feature extraction)**

1. **Anchor-aware HMM topology.** A left-to-right chain of nine core states (first and last
   are anchors) is flanked by two cyclic states that absorb the variable N- and C-terminal
   flanks through self-transitions. Anchor and flank states use specialised emission
   distributions (top-*k* amino acids with pseudocounts).
2. **Ensembles, not single models.** At every node, 50 HMMs are trained from different
   random initialisations; models are ranked by final training log-likelihood and the top
   fraction is used for scoring (median length-normalised log-probability).
3. **Viterbi-embedding split.** Each peptide's Viterbi alignment to the top models gives a
   9-residue core; cores are one-hot encoded, concatenated over models, reduced with PCA
   and split into two branches with K-means (k = 2).
4. **Cross-enrichment.** Both child ensembles rescore all peptides of the parent node; each
   peptide moves to the branch whose HMMs describe it better, and both ensembles are
   retrained. The cycle is repeated 3 times.
5. **Stop rule.** The two child motifs are compared by Jensen-Shannon distance (normalised
   by their information content). If they are too similar, splitting stops and the parent
   ensemble is finalised; branches with too low information content are discarded.
6. **Score matrix.** All peptides are scored against the final ensemble, giving a
   peptide × HMM matrix `S`.

<img width="854" height="251" alt="Снимок экрана — 2026-11-24 в 16 39 57" src="https://github.com/user-attachments/assets/b14cc494-316a-4b46-b920-1da257fba6a4" />

**Stage 2 - DualSimplex (probabilistic clustering)**

`S` is shifted to be strictly positive and factorised as `V ≈ W·H`, where `W` (peptides ×
components) holds soft memberships and `H` (components × HMMs) holds component profiles.
The number of components is estimated from the singular-value spectrum of the matrix.
Details: [How DualSimplex is applied](#how-dualsimplex-is-applied).


Results are compared with **GibbsCluster 2.0** on the same peptides.

## Repository structure

```
Washu_project/
├── 000_make_iedb_dataset.py                 # Step 0: per-allele datasets from the raw IEDB export
├── 001_generate_simulated_motifs.py         # Step 1: synthetic peptide mixtures with known motifs and noise
├── 002_make_per_allele_per_length_split.py  # Step 2: per-length 5-fold train/test split 
├── simulated_main.ipynb                     # Step 3a: hierarchical HMM training, simulated data
├── main_IEDB.ipynb                          # Step 3b: hierarchical HMM training, IEDB data
├── dual_simplex.R                           # Step 4: DualSimplex + comparison with GibbsCluster
```

## Installation

> **Important - custom pomegranate fork required.**
> The code does **not** run on the standard PyPI `pomegranate`. It relies on a customised
> version that adds `DiscreteDistributionAnchor` and `DiscreteDistributionCycle`, based on
> the codebase at <https://github.com/artyomovlab/MHC_predictor>. Install that fork (and
> its required pomegranate build) before installing the remaining dependencies below.

**Python part (HMMs)**

```bash
# 1. Clone
git clone https://github.com/Annaknowshowtocode/Washu_project.git
cd Washu_project

# 2. Create an environment (Python 3.9 recommended for the pomegranate fork)
conda create -n pomegranate python=3.9
conda activate pomegranate

# 3. Install the custom pomegranate fork first (see the MHC_predictor repo for build steps)

# 4. Remaining dependencies
pip install "numpy<1.24" pandas scipy scikit-learn seaborn matplotlib \
            logomaker mhcgnomes mhcnames "dask[dataframe]" umap-learn \
            networkx colorcet tqdm jupyter

# 5. 
pip install torch==2.2.2 transformers==4.57.6 sentencepiece
```

`numpy<1.24` 

**R part (DualSimplex)**

R ≥ 4.2 with the `DualSimplex`, `NMF` and `hNMF` packages (see the
[DualSimplex paper](https://doi.org/10.1186/s13059-025-03910-1) for installation), and
`dplyr tidyr stringr tibble ggplot2 patchwork caret aricode svglite`.

**Baseline**

[GibbsCluster 2.0](https://services.healthtech.dtu.dk/services/GibbsCluster-2.0/)
(web server or standalone) - only needed to reproduce the comparison.


## Pipeline

Steps 0-2 prepare data; step 3 trains the HMM hierarchy and exports the score matrix;
step 4 runs DualSimplex and the benchmark. Use **step 1 for simulated data** (steps 0 and 2
are skipped) and **steps 0, 2 for real IEDB data**.

### `000_make_iedb_dataset.py` - IEDB preprocessing

Builds per-allele binder tables from a raw IEDB export.

- **Input:** `data/mhc_ligand_full.csv` - the full *MHC Ligand* export from the
  [Immune Epitope Database](https://www.iedb.org/) (two-row header).
- **What it does:** keeps linear peptides composed only of the 20 standard amino acids;
  removes three assay groups inconsistent with the rest (EC50 purified-MHC/fluorescence,
  KD purified-MHC/radioactivity, EC50 cellular-MHC/fluorescence, following Sarkizova et
  al.); converts qualitative annotations to nominal affinities (Positive-High 100 nM,
  Positive-Intermediate 1000 nM, Positive-Low 5000 nM, Positive 500 nM, Negative 5000 nM)
  and pools them with nM measurements; labels a peptide as a **binder** when affinity
  ≤ 500 nM; collapses duplicate records; subsets to the alleles in the `ALLELES` list;
  normalises allele names with `mhcgnomes`.
- **Output:** one CSV per allele, `data/simple_model_enrichment/data#<compact_allele>.csv`
  (`;`-separated), plus a dataset-size bar plot.
- **Configure:** `DATA_PATH_BASE`, `EXPERIMENT_NAME`, `ALLELES`.

### `001_generate_simulated_motifs.py` - synthetic data generation

Generates peptide mixtures with a fully controlled motif structure for validation against
a known ground truth.

- **Input:** none.
- **What it does:** builds peptides of 12-18 residues (class II) from a 9-residue core
  with motif-specific restrictions at P1, P4, P6 and P9; core start positions follow a
  discretised Gaussian; flanks are drawn from one shared random amino-acid distribution.
  With probability `--noise-level` (default **25%**) the motif core is replaced by a core
  from an unrestricted position-specific matrix (a separate noise matrix per motif class).
  Up to 5 built-in motifs, optional `--ratios`.
- **Output:** `<output-dir>/sim_data_motifs_<k>_ratio_(...)_lengths_(8,)_noize_<p>.csv`
  (`;`-separated; columns `peptide, core, core_start, core_length, length, motif, allele
  (Dummy_allele_i), noise, binder`) and ground-truth logos under `<output-dir>/logos/`.
  `allele` and `noise` are used for evaluation only, never for training.
- **Configure:**

```bash
python 001_generate_simulated_motifs.py \
    --output-dir data/simulated_data \
    --class-to-generate II \
    --motif-count 5 \
    --noise-level 0.25 \
    --n-peptides 12000 \
    --seed 42
```

### `002_make_per_allele_per_length_split.py` - per-length / k-fold split

Prepares the preprocessed IEDB data for training.

- **Input:** the `data#<allele>.csv` files from step 0.
- **What it does:** keeps binders, groups peptides by length (8-30), and makes a shuffled
  **5-fold** train/test split per length (`sklearn.KFold`; lengths with ≤ 5 peptides are
  skipped). Also renders length histograms and example sequence logos as sanity checks.
- **Output:** `data/simple_model_enrichment/per_length_per_kfold_split/<allele>#binders#train.csv`
  and `…#test.csv` (columns `peptide; length; split`), plus logos under `logos/`.

### `simulated_main.ipynb` / `main_IEDB.ipynb` - hierarchical HMM training

The main drivers of Stage 1. The two notebooks are identical except for the data scenario.

- **Input:** the simulated CSV from step 1 (`SimulatedDataParams`) or the
  `per_length_per_kfold_split/` tables from step 2 (`PreprocessedIEDBDataParams`).
- **What they do:**
  1. load the data with `get_train_test_data`; for IEDB (`collapse_to_dummy=True`) peptides
     shared between alleles are removed, alleles are balanced to equal size preserving the
     length distribution, the true allele is kept as `source_allele` (evaluation only) and
     everything is merged into one unlabelled `dummy_allele`;
  2. keep lengths 12-18, compute length-based weights and export a plain peptide list
     (`peptides_for_gibbscluster.txt`) for the GibbsCluster baseline;
  3. train the **root ensemble** of 50 HMMs (`prepare_multiple_models`,
     `train_multiple_models`) and rank them by score;
  4. run `hierarchically_train_splited_models` - split, cross-enrichment, stop rule (5
     layers for simulated data, 3 for IEDB);
  5. (simulated notebook, optional) K-means / agglomerative / spectral clustering directly
     on the score matrix as additional baselines.
- **Output:** per branch (`br1`, `br11`, `br12`, …) trained models (`model.json`,
  `history.csv`) and, after every layer, in `<result_dir>/diagnostics/<dummy_allele>/`:
  - `layer{N}_scores_matrix.csv` - the peptide × HMM score matrix `S` (mean per-residue
    log-probability of the peptide under each HMM);
  - `layer{N}_clustermap_row_order.csv` - row annotation (`peptide, cluster, allele, noise`);
  - clustermaps and UMAPs (SVG).
- **Configure:** `root_dir`, `num_runs`, `lengths_to_use`, `total_layers_to_do`.

<img width="883" height="422" alt="Снимок экрана — 2026-11-24 в 16 43 17" src="https://github.com/user-attachments/assets/fbd3dc20-82fc-4029-9a22-804669066bf6" />


### `dual_simplex.R` - DualSimplex clustering and benchmark

Stage 2.

- **Input:** `layer{N}_scores_matrix.csv` and `layer{N}_clustermap_row_order.csv` of the
  last layer; optionally GibbsCluster output `gibbs.<k>g.out`.
- **What it does:** factorises the score matrix with DualSimplex, exports memberships and
  component profiles, and benchmarks against GibbsCluster (confusion matrices, accuracy,
  macro precision/recall/F1, ARI, NMI, 1 - NID).
- **Output:** `H_top5_models_per_component.csv`, `top_peptides.csv`, `top_gibbs.csv`,
  UMAP/simplex plots, `clustering_performance_comparison.svg`.
- **Configure:** number of components `K` in `dso$project(K)`, top-*N* peptides `N`,
  file paths.

## How DualSimplex is applied

DualSimplex factorises a non-negative matrix as `V ≈ W·H`. Here `V` is the score matrix
(N peptides × M HMMs), `W` (N × K) are peptide memberships and `H` (K × M) are component
profiles over the HMM ensemble. After a Sinkhorn transformation the data are projected into
a low-dimensional space, where the vertices of two linked simplices (one for peptides, one
for HMMs) are optimised; the factors are read off these vertices.

The script performs the following steps.

1. **Make the matrix strictly positive** with a single global shift that keeps score
   differences:
   ```r
   scores_matrix <- scores_matrix - min(scores_matrix) + 1e-8
   ```
2. **Load into the solver and project.** `K` is chosen from the singular-value spectrum
   (K = 5 for the simulated mixture, K = 3 for the IEDB DRB1 mixture):
   ```r
   dso <- DualSimplexSolver$new()
   dso$set_data(as.matrix(scores_matrix))
   dso$project(K)
   dso$run_umap()
   ```
   True allele and noise labels are attached as annotation only to colour the plots.
3. **Drop uninformative HMMs** (columns with `zero_distance < 0.001`), then re-create the
   solver on the filtered matrix and project again.
4. **Initialise** the simplex vertices from the data:
   ```r
   set.seed(42)
   dso$init_solution("select_omega")
   ```
5. **Optimise** the vertices (5 rounds of 10 iterations), then check the error history:
   ```r
   dso$optim_solution(10, config = optim_config(
       coef_hinge_H = 1, coef_hinge_W = 1,
       coef_der_X = 0.01, coef_der_Omega = 0.01,
       method = "positivity",
       convergence_tol = 1e-30, stop_criteria_window = 1000))
   dso$plot_error_history()
   ```
6. **Finalise** and normalise memberships so each peptide's weights sum to one:
   ```r
   solution <- dso$finalize_solution()
   peptide_to_components <- as.data.frame(solution$W)
   peptide_to_components <- peptide_to_components / rowSums(peptide_to_components)
   ```
   Peptides near a simplex vertex belong to one component, peptides on an edge are shared
   between two, and peptides near the centre (typically noise) have no dominant component,
   so the weights also flag ambiguous assignments.
7. **Export.** The five highest-weight HMMs of every component
   (`H_top5_models_per_component.csv`) and the top-*N* peptides per component by
   normalised membership (`top_peptides.csv`); these peptides are used for the sequence
   logos.
8. **Benchmark against GibbsCluster 2.0.** For GibbsCluster the top-*N* peptides per
   cluster are taken by corrected score (`gibbs.<k>g.out`). For both methods a confusion
   matrix (predicted cluster × true allele) is built; clusters are matched to alleles
   greedily and one-to-one (largest count first), the true noise class is excluded, and
   accuracy, macro precision/recall/F1 (`caret`) and ARI, NMI, 1 - NID (`aricode`) are
   computed.

## Reproducing the experiments

**Simulated data (5 motifs, 25% noise)**

```bash
python 001_generate_simulated_motifs.py -o data/simulated_data --class II \
       --motif-count 5 --noise-level 0.25 --n-peptides 12000 --seed 42
# run simulated_main.ipynb  (set root_dir; total_layers_to_do=5)
# run dual_simplex.R on the last layer with K = 5
# run GibbsCluster 2.0 on peptides_for_gibbscluster.txt and re-run the benchmark part
```

**Real data (IEDB, HLA-DRB1\*01:01 / \*03:01 / \*04:01)**

```bash
python 000_make_iedb_dataset.py                    # needs data/mhc_ligand_full.csv
python 002_make_per_allele_per_length_split.py
# in data_reading_methods.py enable the DRB1 KEEP set
# run main_IEDB.ipynb  (PreprocessedIEDBDataParams, total_layers_to_do=3)
# run dual_simplex.R on the last layer with K = 3
# run GibbsCluster 2.0 (k = 3) on the exported peptide list and re-run the benchmark part
```

For the **HLA-DPA1\*02:02/DPB1\*05:01 sub-motif analysis**, enable the DP `KEEP` set and
repeat the same steps.

A full HMM training run takes on the order of 2 hours on a laptop CPU.

## Modules

| Module | Role |
| ------ | ---- |
| **`training_parameters.py`** | All configuration classes: data scenarios (`SimulatedDataParams`, `PreprocessedIEDBDataParams`, …), `ModelTrainingParams`, the ready-to-use `SimpleModelClassIIParams`, and `ExperimentParams` tying name, data and result paths together. |
| **`data_reading_methods.py`** | Reads per-allele / k-fold / length data, removes peptides shared between alleles, balances alleles preserving length distributions, collapses alleles into one dummy mixture, computes length-based sample weights. |
| **`peptides_utils.py`** | Small helpers: MHC class detection, sequence-logo creation, dict split/join, background frequencies. |
| **`hmm_model_params.py`** | Presets of HMM architecture, pseudocounts and anchor/cycle settings. |
| **`hmm_logic_methods.py`** | The engine. Builds HMMs from parameters (`build_model_based_on_params`), trains ensembles, runs hierarchical splitting with cross-enrichment (`make_single_split_and_enrich`, `hierarchically_train_splited_models`), Viterbi-embedding + PCA + K-means split, stop rule, score-matrix export. |
| **`core_embedding_split.py`** | Optional alternative split: embeds extracted cores with a protein language model (ESM-2, ProtBERT, ProtT5) before PCA + K-means. |
| **`hmm_visualization_methods.py`** | Sequence logos, clustermaps, UMAP projections, per-layer split diagnostics. |
| **`hmm_metrics.py`** | Loads reference PSSMs (`.mat`), information content, KL divergence from background, correlation metrics. |
| **`motif_benchmark.py`** | Aligns HMM branches and compares recovered motifs with reference matrices. |

## Configuration

Key knobs live in `ModelTrainingParams` / `SimpleModelClassIIParams`
(`training_parameters.py`) and `hmm_model_params.py`:

| Parameter | Used value | Meaning |
| --------- | ---------- | ------- |
| `num_runs` | `50` | HMMs per ensemble |
| `lengths_to_use` | `12-18` | peptide lengths included |
| `enrichment_steps` | `3` | branch-reassignment iterations per split |
| `total_layers_to_do` | `5` (simulated), `3` (IEDB) | depth of the hierarchy |
| `top_n_models` (split) | `25` (top 50% of the ensemble) | models used for Viterbi embedding |
| PCA | 95% variance, ≤ 30 components | dimensionality reduction before K-means |
| K-means | `k = 2`, fitted on split 0 | branch split |
| `trash_model_information_threshold` | see `training_parameters.py` | minimum information content to keep a branch |
| `same_model_distance_threshold` | see `training_parameters.py` | minimum normalised JS distance to keep branches separate |
| `emission_pseudocount`, `transition_pseudocount`, `anchor_top_aas`, … | see `hmm_model_params.py` | HMM regularisation |
| DualSimplex `K` | `5` (simulated), `3` (IEDB) | number of components |

## Outputs

A completed run produces:

- **Trained HMMs** per branch (`model.json`, `history.csv`).
- **Peptide × HMM score matrices** per layer (`layer{N}_scores_matrix.csv`) with
  clustermaps and UMAP projections coloured by true allele and by cluster.
- **DualSimplex solution:** peptide memberships `W`, component profiles `H`, top HMMs per
  component, top-*N* peptides per component for sequence logos.
- **Benchmark tables and figures** against GibbsCluster 2.0 (confusion matrices,
  classification and clustering metrics).

## References & acknowledgements

- Klimova A., Kleverov D., Shalyto A., Artyomov M. *Clustering peptide sequences using
  Hidden Markov Models and matrix factorization methods.*
- The HMM component is based on and extends the interpretable HMM framework of
  Kleverov D.A., Shalyto A.A., Artyomov M. *A method for constructing interpretable hidden
  Markov models for the task of finding peptide binding sites in protein sequences.*
  Scientific and Technical Journal of Information Technologies, Mechanics and Optics,
  2023, 23(5), 989-1000. DOI: https://doi.org/10.17586/2226-1494-2023-23-5-989-1000
- Polezhaeva V.A., Kleverov D.A., Shalyto A.A., Artyomov M. *Incorporating negative
  examples into Hidden Markov Model-based classification of peptide sequences.* Scientific
  and Technical Journal of Information Technologies, Mechanics and Optics, 2025, 25(5),
  888-901. DOI: https://doi.org/10.17586/2226-1494-2025-25-5-888-901
- DualSimplex: Kleverov D., Aladyeva E., Serdyukov A., Artyomov M.N. *Non-negative matrix
  factorization and deconvolution as a dual simplex problem.* Genome Biology, 2026, 27(1),
  25. DOI: https://doi.org/10.1186/s13059-025-03910-1
- - Built on top of the HMM codebase at
  [`denklewer/MHC_predictor`](https://github.com/denklewer/MHC_predictor) and a customised
  [pomegranate](https://github.com/denklewer/pomegranate/tree/before_py_torch) build
  (branch `before_py_torch`).
- GibbsCluster: Andreatta M., Alvarez B., Nielsen M., *Nucleic Acids Res.* 2017, 45(W1),
  W458-W463.
- Assay filtering and binder definition: Sarkizova S. et al., *Nat. Biotechnol.* 2020,
  38(2), 199-209.
- Data: the [Immune Epitope Database (IEDB)](https://www.iedb.org/).

Contacts: mus.anna.study@gmail.com · denklewer@gmail.com
