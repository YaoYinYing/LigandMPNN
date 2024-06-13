# LigandMPNN

This package provides inference code for [LigandMPNN](https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1) & [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) models. The code and model parameters are available under the MIT license.

Third party code: side chain packing uses helper functions from [Openfold](https://github.com/aqlaboratory/openfold).

---
**Note**: This fork is a modified version of the original LigandMPNN code.

## Main Feature of this fork

1. A pip-installable package for dependency management:
2. Using `hydra` for config management: `ligandmpnn/config/ligandmpnn.yaml`.
3. Commandline interfaces for inferences and pretrained weight fetching:
4. Download pretrained weight file automaticaly while it is used.
5. Slightly deduplicate `run.py`/`score.py` without changing the orginal libraries (`*_utils`)
6. Sidechain solving (using `openfold`, installed via pip+git).
7. Customized checkpoint file/url
8. Scoring from given sequence(experimental feature).
9. Force to use CPU even CUDA is available.
10. CI tests:

- All test cases are passing in CI runners (py3.9-3.11, ubuntu): <https://github.com/YaoYinYing/LigandMPNN/actions/runs/8519719844>
- All tests w/o sidechain modeling passed on ubuntu/Windows/MacOS(M1&Intel): <https://github.com/YaoYinYing/LigandMPNN/actions/runs/8793510182>
- Windows tests on sidechain failed due to OpenFold code.
- For detailed examples, ckeck example scripts: `run_examples.sh`, `sc_examples.sh`, `score_examples.sh`
- Latest CI: [status](https://github.com/YaoYinYing/LigandMPNN/actions/workflows/unit_tests_tag.yml)

## Running the code

### Installation

```shell
# from github
pip install git+https://github.com/YaoYinYing/LigandMPNN
pip install -e  'git+https://github.com/YaoYinYing/LigandMPNN#egg=ligandmpnn[openfold]'

# or from cloned repo
pip install .
pip install .[openfold]
```

### Getting started

```shell
# design
ligandmpnn input.pdb="./inputs/1BC8.pdb" output.folder="./test/default"

# scoring
ligandmpnn runtime.mode.use='score' model_type.use='ligand_mpnn' input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" output.folder="./test/scorer" scorer.use_sequence=False sampling.number_of_batches=10 runtime.force_cpu=True
```

## Main differences compared with [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) code

- Input PDBs are parsed using [Prody](https://pypi.org/project/ProDy/) preserving protein residue indices, chain letters, and insertion codes. If there are missing residues in the input structure the output fasta file won't have added `X` to fill the gaps. The script outputs .fasta and .pdb files. It's recommended to use .pdb files since they will hold information about chain letters and residue indices.
- Adding bias, fixing residues, and selecting residues to be redesigned now can be done using residue indices directly, e.g. A23 (means chain A residue with index 23), B42D (chain B, residue 42, insertion code D).
- Model writes to fasta files: `overall_confidence`, `ligand_confidence` which reflect the average confidence/probability (with T=1.0) over the redesigned residues  `overall_confidence=exp[-mean_over_residues(log_probs)]`. Higher numbers mean the model is more confident about that sequence. min_value=0.0; max_value=1.0. Sequence recovery with respect to the input sequence is calculated only over the redesigned residues.

### Model parameters

To download model parameters run:

```shell
# fetching all weights
ligandmpnn_download_weights /mnt/db/weights/ligandmpnn/
```

Optionally, you can download only the weights you need while running the model. This is an automatic process.

```shell
ligandmpnn input.pdb="./inputs/1BC8.pdb" output.folder="./test/default" model_type.use='ligand_mpnn' checkpoint.ligand_mpnn.use='ligandmpnn_v_32_020_25'
Seed: 42
Device:cpu: None
Downloading data from 'https://files.ipd.uw.edu/pub/ligandmpnn//ligandmpnn_v_32_020_25.pt' to file '/Users/yyy/Documents/protein_design/LigandMPNN/model_params/ligandmpnn_v_32_020_25.pt'.
100%|█████████████████████████████████████| 10.5M/10.5M [00:00<00:00, 21.0GB/s]
Mode: design
Designing protein from this path: ./inputs/1BC8.pdb
[2024-03-24 01:10:35,668][.prody][DEBUG] - 1356 atoms and 1 coordinate set(s) were parsed in 0.01s.
These residues will be redesigned:  ...
```

## Available models

To run the model of your choice specify `--model_type` and optionally the model checkpoint path. Available models:

- ProteinMPNN

```text
proteinmpnn_v_48_002.pt" #noised with 0.02A Gaussian noise
proteinmpnn_v_48_010.pt" #noised with 0.10A Gaussian noise
proteinmpnn_v_48_020.pt" #noised with 0.20A Gaussian noise
proteinmpnn_v_48_030.pt" #noised with 0.30A Gaussian noise
```

- LigandMPNN

```text
ligandmpnn_v_32_005_25.pt" #noised with 0.05A Gaussian noise
ligandmpnn_v_32_010_25.pt" #noised with 0.10A Gaussian noise
ligandmpnn_v_32_020_25.pt" #noised with 0.20A Gaussian noise
ligandmpnn_v_32_030_25.pt" #noised with 0.30A Gaussian noise
```

- SolubleMPNN

```text
solublempnn_v_48_002.pt" #noised with 0.02A Gaussian noise
solublempnn_v_48_010.pt" #noised with 0.10A Gaussian noise
solublempnn_v_48_020.pt" #noised with 0.20A Gaussian noise
solublempnn_v_48_030.pt" #noised with 0.30A Gaussian noise
```

- ProteinMPNN with global membrane label

```text
global_label_membrane_mpnn_v_48_020.pt" #noised with 0.20A Gaussian noise
```

- ProteinMPNN with per residue membrane label

```text
per_residue_label_membrane_mpnn_v_48_020.pt" #noised with 0.20A Gaussian noise
```

- Side chain packing model

```text
ligandmpnn_sc_v_32_002_16.pt"
```

## Design examples

### 1 default

Default settings will run ProteinMPNN.

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/default"
```

### 2 sampling.temperature

`sampling.temperature=0.05` Change sampling temperature (higher temperature gives more sequence diversity).

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        sampling.temperature=0.05 \
        output.folder="./outputs/temperature"
```

### 3 sampling.seed

`sampling.seed=111` Not selecting a seed will run with a random seed. Running this multiple times will give different results.

```shell
ligandmpnn \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/random_seed"
```

### 4 runtime.verbose

`runtime.verbose=False` Do not print any statements.

```shell
ligandmpnn \
        sampling.seed=111 \
        runtime.verbose=False \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/verbose"
```

### 5 output.save_stats

`output.save_stats=True` Save sequence design statistics.

```shell
#['generated_sequences', 'sampling_probs', 'log_probs', 'decoding_order', 'native_sequence', 'mask', 'chain_mask', 'seed', 'temperature']
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/save_stats" \
        output.save_stats=True
```

### 6 input.fixed_residues

`input.fixed_residues` Fixing specific amino acids. This example fixes the first 10 residues in chain C and adds global bias towards A (alanine). The output should have all alanines except the first 10 residues should be the same as in the input sequence since those are fixed.

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/fix_residues" \
        input.fixed_residues="C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" \
        input.bias.bias_AA="A:10.0"
```

### 7 input.redesigned_residues

`input.redesigned_residues` Specifying which residues need to be designed. This example redesigns the first 10 residues while fixing everything else.

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/redesign_residues" \
        input.redesigned_residues="C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" \
        input.bias.bias_AA="A:10.0"
```

### 8 sampling.number_of_batches

Design 15 sequences; with batch size 3 (can be 1 when using CPUs) and the number of batches 5.

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/batch_size" \
        sampling.batch_size=3 \
        sampling.number_of_batches=5
```

### 9 input.bias.bias_AA

Global amino acid bias. In this example, output sequences are biased towards W, P, C and away from A.

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        input.bias.bias_AA=\"W:3.0,P:3.0,C:3.0,A:-3.0\" \
        output.folder="./outputs/global_bias"
```

### 10 input.bias.bias_AA_per_residue

Specify per residue amino acid bias, e.g. make residues C1, C3, C5, and C7 to be prolines.

```shell
# {
# "C1": {"G": -0.3, "C": -2.0, "P": 10.8},
# "C3": {"P": 10.0},
# "C5": {"G": -1.3, "P": 10.0},
# "C7": {"G": -1.3, "P": 10.0}
# }
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        input.bias.bias_AA_per_residue="./inputs/bias_AA_per_residue.json" \
        output.folder="./outputs/per_residue_bias"
```

### 11 input.bias.omit_AA

Global amino acid restrictions. This is equivalent to using `--bias_AA` and setting bias to be a large negative number. The output should be just made of E, K, A.

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        input.bias.omit_AA="CDFGHILMNPQRSTVWY" \
        output.folder="./outputs/global_omit"
```

### 12 input.bias.omit_AA_per_residue

Per residue amino acid restrictions.

```shell
# {
# "C1": "ACDEFGHIKLMNPQRSTVW",
# "C3": "ACDEFGHIKLMNPQRSTVW",
# "C5": "ACDEFGHIKLMNPQRSTVW",
# "C7": "ACDEFGHIKLMNPQRSTVW"
# }
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        input.bias.omit_AA_per_residue="./inputs/omit_AA_per_residue.json" \
        output.folder="./outputs/per_residue_omit"
```

### 13 input.symmetry.symmetry_weights

Designing sequences with symmetry, e.g. homooligomer/2-state proteins, etc. In this example make C1=C2=C3, also C4=C5, and C6=C7.

```shell
#total_logits += symmetry_weights[t]*logits
#probs = torch.nn.functional.softmax((total_logits+bias_t) / temperature, dim=-1)
#total_logits_123 = 0.33*logits_1+0.33*logits_2+0.33*logits_3
#output should be ***ooxx
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/symmetry" \
        input.symmetry.symmetry_residues=\"C1,C2,C3+C4,C5+C6,C7\" \
        input.symmetry.symmetry_weights=\"0.33,0.33,0.33+0.5,0.5+0.5,0.5\"

```

### 14 input.symmetry.homo_oligomer

Design homooligomer sequences. This automatically sets `input.symmetry.symmetry_residues` and `input.symmetry.symmetry_weights` assuming equal weighting from all chains.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/4GYT.pdb" \
        output.folder="./outputs/homooligomer" \
        input.symmetry.homo_oligomer=True \
        sampling.number_of_batches=2
```

### 15 output.file_ending

Outputs will have a specified ending; e.g. `1BC8_xyz.fa` instead of `1BC8.fa`

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/file_ending" \
        output.file_ending="_xyz"
```

### 16 output.zero_indexed

Zero indexed names in /backbones/1BC8_0.pdb, 1BC8_1.pdb, 1BC8_2.pdb etc

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/zero_indexed" \
        output.zero_indexed=True \
        sampling.number_of_batches=2
```

### 17 input.chains_to_design

Specify which chains (e.g. "A,B,C") need to be redesigned, other chains will be kept fixed. Outputs in seqs/backbones will still have atoms/sequences for the whole input PDB.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/4GYT.pdb" \
        output.folder="./outputs/chains_to_design" \
        input.chains_to_design=[A,B]
```

### 18 input.parse_these_chains_only

Parse and design only specified chains (e.g. "A,B,C"). Outputs will have only specified chains.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/4GYT.pdb" \
        output.folder="./outputs/parse_these_chains_only" \
        input.parse_these_chains_only=[A,B]
```

### 19 model_type.use="ligand_mpnn"

Run LigandMPNN with default settings.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/ligandmpnn_default""
```

### 20 checkpoint.ligand_mpnn.use

Run LigandMPNN using 0.05A model by specifying `checkpoint.ligand_mpnn.use` flag.

```shell
ligandmpnn \
        checkpoint.ligand_mpnn.use="ligandmpnn_v_32_005_25" \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/ligandmpnn_v_32_005_25"
```

### 21 sampling.ligand_mpnn.use_atom_context

Setting `sampling.ligand_mpnn.use_atom_context=False` will mask all ligand atoms. This can be used to assess how much ligand atoms affect AA probabilities.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/ligandmpnn_no_context" \
        sampling.ligand_mpnn.use_atom_context=False 
```

### 22 sampling.ligand_mpnn.use_side_chain_context

Use fixed residue side chain atoms as extra ligand atoms.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/ligandmpnn_use_side_chain_atoms" \
        sampling.ligand_mpnn.use_side_chain_context=True \
        input.fixed_residues="C1 C2 C3 C4 C5 C6 C7 C8 C9 C10"
```

### 23 model_type.use="soluble_mpnn"

Run SolubleMPNN (ProteinMPNN-like model with only soluble proteins in the training dataset).

```shell
ligandmpnn \
        model_type.use="soluble_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/soluble_mpnn_default"
```

### 24 model_type.use="global_label_membrane_mpnn"

Run global label membrane MPNN (trained with extra input - binary label soluble vs not) `input.transmembrane.global_transmembrane_label #True - membrane, False - soluble`.

```shell
ligandmpnn \
        model_type.use="global_label_membrane_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/global_label_membrane_mpnn_0" \
        input.transmembrane.global_transmembrane_label=False 
```

### 25 model_type.use="per_residue_label_membrane_mpnn"

Run per residue label membrane MPNN (trained with extra input per residue specifying buried (hydrophobic), interface (polar), or other type residues; 3 classes).

```shell
ligandmpnn \
        model_type.use="per_residue_label_membrane_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/per_residue_label_membrane_mpnn_default" \
        input.transmembrane.buried="C1 C2 C3 C11" \
        input.transmembrane.interface="C4 C5 C6 C22"
```

### 26 output.fasta_seq_separation

Choose a symbol to put between different chains in fasta output format. It's recommended to PDB output format to deal with residue jumps and multiple chain parsing.

```shell
ligandmpnn \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/fasta_seq_separation" \
        output.fasta_seq_separation=":"
```

### 27 input.pdb_path_multi

Specify multiple PDB input paths. This is more efficient since the model needs to be loaded from the checkpoint once.

```shell
#{
#"./inputs/1BC8.pdb": "",
#"./inputs/4GYT.pdb": ""
#}
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        output.folder="./outputs/pdb_path_multi" \
        sampling.seed=111
```

### 28 input.fixed_residues_multi

Specify fixed residues when using `input.fixed_residues_multi` flag.

```shell
#{
#"./inputs/1BC8.pdb": "C1 C2 C3 C4 C5 C10 C22",
#"./inputs/4GYT.pdb": "A7 A8 A9 A10 A11 A12 A13 B38"
#}
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        input.fixed_residues_multi="./inputs/fix_residues_multi.json" \
        output.folder="./outputs/fixed_residues_multi" \
        sampling.seed=111
```

### 29 input.redesigned_residues_multi

Specify which residues need to be redesigned when using `input.redesigned_residues_multi` flag.

```shell
#{
#"./inputs/1BC8.pdb": "C1 C2 C3 C4 C5 C10",
#"./inputs/4GYT.pdb": "A7 A8 A9 A10 A12 A13 B38"
#}
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        input.redesigned_residues_multi="./inputs/redesigned_residues_multi.json" \
        output.folder="./outputs/redesigned_residues_multi" \
        sampling.seed=111

```

### 30 input.bias.omit_AA_per_residue_multi

Specify which residues need to be omitted when using `input.bias.omit_AA_per_residue_multi` flag.

```shell
#{
#"./inputs/1BC8.pdb": {"C1":"ACDEFGHILMNPQRSTVWY", "C2":"ACDEFGHILMNPQRSTVWY", "C3":"ACDEFGHILMNPQRSTVWY"},
#"./inputs/4GYT.pdb": {"A7":"ACDEFGHILMNPQRSTVWY", "A8":"ACDEFGHILMNPQRSTVWY"}
#}
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        input.bias.omit_AA_per_residue_multi="./inputs/omit_AA_per_residue_multi.json" \
        output.folder="./outputs/omit_AA_per_residue_multi" \
        sampling.seed=111
```

### 31 input.bias.bias_AA_per_residue_multi

Specify amino acid biases per residue when using `input.bias.bias_AA_per_residue_multi` flag.

```shell
#{
#"./inputs/1BC8.pdb": {"C1":{"A":3.0, "P":-2.0}, "C2":{"W":10.0, "G":-0.43}},
#"./inputs/4GYT.pdb": {"A7":{"Y":5.0, "S":-2.0}, "A8":{"M":3.9, "G":-0.43}}
#}
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        input.bias.bias_AA_per_residue_multi="./inputs/bias_AA_per_residue_multi.json" \
        output.folder="./outputs/bias_AA_per_residue_multi" \
        sampling.seed=111
```

### 32 sampling.ligand_mpnn.cutoff_for_score

This sets the cutoff distance in angstroms to select residues that are considered to be close to ligand atoms. This flag only affects the `num_ligand_res` and `ligand_confidence` in the output fasta files.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        sampling.ligand_mpnn.cutoff_for_score="6.0" \
        output.folder="./outputs/ligand_mpnn_cutoff_for_score"
```

### 33 specifying residues with insertion codes

You can specify residue using chain_id + residue_number + insersion_code; e.g. redesign only residue B82, B82A, B82B, B82C.

```shell
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/2GFB.pdb" \
        output.folder="./outputs/insertion_code" \
        input.redesigned_residues="B82 B82A B82B B82C" \
        input.parse_these_chains_only="B"
```

### 34 parse atoms with zero occupancy

Parse atoms in the PDB files with zero occupancy too.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/parse_atoms_with_zero_occupancy" \
        input.parse_atoms_with_zero_occupancy=True
```

### 35 Customized checkpoints from file/url(md5sum is optional)

```shell
mkdir -p customized_weight_dir_local
curl 'https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt' -o customized_weight_dir_local/customized_proteinmpnn_v_48_002.pt
ls customized_weight_dir_local
ligandmpnn \
        sampling.seed=111 \
        weight_dir="customized_weight_dir_local" \
        checkpoint.customized.file=customized_weight_dir_local/customized_proteinmpnn_v_48_002.pt \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/default_customozed_weight_local"



mkdir -p customized_weight_dir_remote
ligandmpnn \
        sampling.seed=111 \
        weight_dir="customized_weight_dir_remote" \
        checkpoint.customized.url='https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt' \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/customized_weight_dir_remote"

ls customized_weight_dir_remote



mkdir -p customized_weight_dir_remote_hash
ligandmpnn \
        sampling.seed=111 \
        weight_dir="customized_weight_dir_remote_hash" \
        checkpoint.customized.url='https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt' \
        checkpoint.customized.known_hash='md5:4255760493a761d2b6cb0671a48e49b7' \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/customized_weight_dir_remote_hash"

ls customized_weight_dir_remote_hash
```

## Scoring examples

### Output dictionary

- ["logits"] - raw logits from the model
- ["probs"] - softmax(logits)
- ["log_probs"] - log_softmax(logits)
- ["decoding_order"] - decoding order used (logits will depend on the decoding order)
- ["native_sequence"] - parsed input sequence in integers
- ["mask"] - mask for missing residues (usually all ones)
- ["chain_mask"] - controls which residues are decoded first
- ["alphabet"] - amino acid alphabet used
- ["residue_names"] - dictionary to map integers to residue_names, e.g. {0: "C10", 1: "C11"}
- ["sequence"] - parsed input sequence in alphabet
- ["mean_of_probs"] - averaged over batch_size*number_of_batches probabilities, [protein_length, 21]
- ["std_of_probs"] - same as above, but std

### 1 autoregressive with sequence info

Get probabilities/scores for backbone-sequence pairs using autoregressive probabilities: p(AA_1|backbone), p(AA_2|backbone, AA_1) etc. These probabilities will depend on the decoding order, so it's recomended to set number_of_batches to at least 10.

```shell
ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/autoregressive_score_w_seq" \
        scorer.use_sequence=True \
        sampling.batch_size=1 \
        sampling.number_of_batches=10
```

### 2 autoregressive with backbone info only

Get probabilities/scores for backbone using probabilities: p(AA_1|backbone), p(AA_2|backbone) etc. These probabilities will depend on the decoding order, so it's recomended to set number_of_batches to at least 10.

```shell
ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/autoregressive_score_wo_seq" \
        scorer.use_sequence=False \
        sampling.batch_size=1 \
        sampling.number_of_batches=10

```

### 3 single amino acid score with sequence info

Get probabilities/scores for backbone-sequence pairs using single aa probabilities: p(AA_1|backbone, AA_{all except AA_1}), p(AA_2|backbone, AA_{all except AA_2}) etc. These probabilities will depend on the decoding order, so it's recomended to set number_of_batches to at least 10.

```shell
ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.single_aa_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/single_aa_score_w_seq" \
        scorer.use_sequence=True \
        sampling.batch_size=1 \
        sampling.number_of_batches=10
```

### 4 single amino acid score with backbone info only

Get probabilities/scores for backbone-sequence pairs using single aa probabilities: p(AA_1|backbone), p(AA_2|backbone) etc. These probabilities will depend on the decoding order, so it's recomended to set number_of_batches to at least 10.

```shell
ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.single_aa_score=True \
        input.pdb="./outputs/ligandmpnn_default/backbones/1BC8_1.pdb" \
        output.folder="./outputs/single_aa_score_wo_seq" \
        scorer.use_sequence=False \
        sampling.batch_size=1 \
        sampling.number_of_batches=10
```

### score from given sequence (**Not validated**)

```shell
ligandmpnn \
        runtime.mode.use='score' \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        scorer.autoregressive_score=True \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/autoregressive_score_w_seq_fasta_bz10" \
        scorer.use_sequence=True \
        sampling.batch_size=10 \
        sampling.number_of_batches=1 \
        scorer.customized_seq='GMSSISLPEFLLELLSDPKYEDYIKWVSDNGEFELKNPEAVAKLWGEKKGLPDMNYEKMYKELKKYEKKKIIEKVKGKPNVYKFVNYPEILNP' > autoregressive_score_w_seq_fasta_bz10.log

```

## Side chain packing examples

### 1 design a new sequence and pack side chains (return 1 side chain packing sample - fast)

Design a new sequence using any of the available models and also pack side chains of the new sequence. Return only a single solution for the side chain packing.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_default_fast" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=False \
        packer.pack_with_ligand_context=True
```

### 2 design a new sequence and pack side chains (return 4 side chain packing samples)

Same as above, but returns 4 independent samples for side chains. b-factor shows log prob density per chi angle group.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_default" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=4 \
        packer.pack_with_ligand_context=True

```

### 3 fix specific residues fors sequence design and packing

This option will not repack side chains of the fixed residues, but use them as a context.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_fixed_residues" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=4 \
        packer.pack_with_ligand_context=True \
        input.fixed_residues="C6 C7 C8 C9 C10 C11 C12 C13 C14 C15" \
        packer.repack_everything=False
```

### 4 fix specific residues for sequence design but repack everything

This option will repacks all the residues.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_fixed_residues_full_repack" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=4 \
        packer.pack_with_ligand_context=True \
        input.fixed_residues="C6 C7 C8 C9 C10 C11 C12 C13 C14 C15" \
        packer.repack_everything=True
```

### 5 design a new sequence using LigandMPNN but pack side chains without considering ligand/DNA etc atoms

You can run side chain packing without taking into account context atoms like DNA atoms. This most likely will results in side chain clashing with context atoms, but it might be interesting to see how model's uncertainty changes when ligand atoms are present vs not for side chain conformations.

```shell
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_no_context" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=4 \
        packer.pack_with_ligand_context=False
```

### Things to add

- Support for ProteinMPNN CA-only model.
- Examples for scoring sequences only.
- Side-chain packing scripts.
- TER

### Citing this work

If you use the code, please cite:

```bibtex
@article{dauparas2023atomic,
  title={Atomic context-conditioned protein sequence design using LigandMPNN},
  author={Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and An, Linna and Anishchenko, Ivan and Glasscock, Cameron and Baker, David},
  journal={Biorxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},  
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```
