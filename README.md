# Getting started
1. Get uv 

curl -LsSf https://astral.sh/uv/install.sh | sh

2. get python 3.11.11

3. create a virtual environment

uv venv --python python3.11.11 .venv
source .venv/bin/activate

then simply 

uv sync --active

4. set docker
sudo apt update
sudo apt upgrade -y

curl -fsSL https://get.docker.com | sudo sh


docker run --gpus all -p 8000:8000 myimage

add yourself to the group 
sudo usermod -aG docker $USER
## Agent
# Agents

export MBX_NO_DOCKER=1

<!-- (ITS-bench) asim@omen:~/Desktop/ITS-bench$ docker run -it mlebench-env:latest /bin/bash
+ tee /entrypoint.log
+ LOGS_DIR=/home/logs
+ mkdir -p /home/logs
+ find /home -path /home/data -prune -o -exec chmod a+rw '{}' ';'
+ ls -l /home
total 28
-rw-rw-rw- 1 root    root    2789 Feb 25 18:26 instructions.txt
-rw-rw-rw- 1 root    root    2586 Feb 25 18:26 instructions_obfuscated.txt
drwxrwxrwx 2 root    root    4096 Apr 30 10:36 logs
drwxrwxrwx 1 nonroot nonroot 4096 Mar  9 16:02 nonroot
drwxrwxrwx 1 root    root    4096 Mar  9 16:02 submission
-rw-rw-rw- 1 root    root     494 Feb 25 18:26 validate_submission.sh
+ /opt/conda/bin/python /private/grading_server.py
 * Serving Flask app 'grading_server'
 * Debug mode: off -->


## Prerequisites
If you want to run these agents locally:
- Install [Docker](https://docs.docker.com/engine/install/)
- Install [Sysbox](https://github.com/nestybox/sysbox). See [Security](#Security) below for more information
- (Optional) Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to run agents with GPUs

To build an image for an agent with ID `<agent>`, run:

```bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

docker build --platform=linux/amd64 -t aide agents/aide-deepseek/ --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR
```
## build the new agent
```
docker build --platform=linux/amd64 -t aide-deepseek agents/aide-deepseek/ --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR
```

## Running agents

Our `run_agent.py` script allows you to run agents locally on a given set of competitions. In the `experiments/splits/` directory, we have several files, each containing a set of competition IDs. The `experiments/splits/all.txt` file contains all competitions. The `experiments/splits/spaceship-titanic.txt` split just contains the Spaceship Titanic competition, which is useful for testing. For example, to run the dummy agent on the Spaceship Titanic competition, you can run:

```bash 
python run_agent.py --agent-id aide --competition-set experiments/splits/chosen_competetions.txt --data-dir Dataset
```

Running `run_agent.py` will creates a "run group" directory in the `runs/` directory. The run group directory will contain a subdirectory for each competition that the agent was evaluated on, containing the agent's logs, code, and submission. A `metadata.json` file will be created on finish within the run group directory, summarizing the results of the runs. You can then grade this run using the `metadata.json` file. For example, to grade the run group `<run-group>`, you can first use `experiments/make_submission.py` to generate a submission JSONL file:

```bash
python experiments/make_submission.py --metadata runs/<run-group>/metadata.json --output runs/<run-group>/submission.jsonl
```

You can then use the `mlebench grade` command to grade this submission:

```bash
mlebench grade --submission runs/<run-group>/submission.jsonl --output-dir runs/<run-group>
```

If you'd like to update the configuration of the container, you can edit the default container config in `environment/config/container_configs/default.json`, or specify a custom container config JSON file when executing `run_agent.py`. If you'd like to run the agent with a GPU, you can set `"gpus": -1` in the container config JSON file.

## Benchmarking

This section describes a canonical setup for comparing scores on MLE-bench. We recommend the following:
- Repeat each evaluation with at least 3 seeds and report the Any Medal (%) score as the mean ± one standard error of the mean. The evaluation (task and grading) itself is deterministic, but agents/LLMs can be quite high-variance!
- Agent resources - not a strict requirement of the benchmark but please report if you stray from these defaults!
  - Runtime: 24 hours
  - Compute: 36 vCPUs with 440GB RAM and one 24GB A10 GPU
- Include a breakdown of your scores across Low, Medium, High, and All complexity [splits](experiments/splits) (see *Lite evaluation* below for why this is useful).

We demonstrate how this looks in practice by reporting the main results from [our paper (Table 2)](https://arxiv.org/abs/2410.07095) in the table below:


### Lite Evaluation

Evaluating agents with the above settings on the full 75 competitions of MLE-bench can be expensive. For users preferring a "lite" version of the benchmark, we recommend using the [Low complexity split](https://github.com/openai/mle-bench/blob/main/experiments/splits/low.txt) of our dataset, which consists of only 22 competitions. This reduces the number of runs substantially, while still allowing fair comparison along one column of the table above.

Furthermore, the Low complexity competitions tend to be significantly more lightweight (158GB total dataset size compared to 3.3TB for the full set), so users may additionally consider reducing the runtime or compute resources available to the agents for further cost reduction. However, note that doing so risks degrading the performance of your agent. For example, see [Section 3.3 and 3.4 of our paper](https://arxiv.org/abs/2410.07095) where we have experimented with varying resources on the full competition set.

The Lite dataset contains the following competitions:

| Competition ID                              | Category                   | Dataset Size (GB) |
|---------------------------------------------|----------------------------|--------------------|
| aerial-cactus-identification                | Image Classification       | 0.0254            |
| aptos2019-blindness-detection               | Image Classification       | 10.22             |
| denoising-dirty-documents                   | Image To Image             | 0.06              |
| detecting-insults-in-social-commentary      | Text Classification        | 0.002             |
| dog-breed-identification                    | Image Classification       | 0.75              |
| dogs-vs-cats-redux-kernels-edition          | Image Classification       | 0.85              |
| histopathologic-cancer-detection            | Image Regression           | 7.76              |
| jigsaw-toxic-comment-classification-challenge | Text Classification        | 0.06              |
| leaf-classification                         | Image Classification       | 0.036             |
| mlsp-2013-birds                             | Audio Classification       | 0.5851            |
| new-york-city-taxi-fare-prediction          | Tabular                   | 5.7               |
| nomad2018-predict-transparent-conductors    | Tabular                   | 0.00624           |
| plant-pathology-2020-fgvc7                  | Image Classification       | 0.8               |
| random-acts-of-pizza                        | Text Classification        | 0.003             |
| ranzcr-clip-catheter-line-classification    | Image Classification       | 13.13             |
| siim-isic-melanoma-classification           | Image Classification       | 116.16            |
| spooky-author-identification                | Text Classification        | 0.0019            |
| tabular-playground-series-dec-2021          | Tabular                   | 0.7               |
| tabular-playground-series-may-2022          | Tabular                   | 0.57              |
| text-normalization-challenge-english-language | Seq->Seq                 | 0.01              |
| text-normalization-challenge-russian-language | Seq->Seq                 | 0.01              |
| the-icml-2013-whale-challenge-right-whale-redux | Audio Classification     | 0.29314           |

## Setup

Some MLE-bench competition data is stored using [Git-LFS](https://git-lfs.com/).
Once you have downloaded and installed LFS, run:

```console
git lfs fetch --all
git lfs pull
```

You can install `mlebench` with pip:

```console
pip install -e .
```

### Pre-Commit Hooks (Optional)

If you're committing code, you can install the pre-commit hooks by running:

```console
pre-commit install
```

## Dataset

We use the [Kaggle API](https://github.com/Kaggle/kaggle-api) to download the
raw datasets. Ensure that you have downloaded your Kaggle credentials
(`kaggle.json`) and placed it in the `~/.kaggle/` directory (this is the default
location where the Kaggle API looks for your credentials). To download and prepare the MLE-bench dataset, run the following, which will download and prepare the dataset in your system's default cache directory. Note, we've found this to take two days when running from scratch:

```console
mlebench prepare --all --data-dir all_datasets
```

To prepare the lite dataset, run:

```console
mlebench prepare --lite --data-dir lite_dataset
```

Alternatively, you can prepare the dataset for a specific competition by
running:

```console
mlebench prepare -c <competition-id> --data-dir .
```

Run `mlebench prepare --help` to see the list of available competitions.



## Grading Submissions

Answers for competitions must be submitted in CSV format; the required format is described in each competition's description, or shown in a competition's sample submission file. You can grade multiple submissions by using the `mlebench grade` command. Given a JSONL file, where each line corresponds with a submission for one competition, `mlebench grade` will produce a grading report for each competition. The JSONL file must contain the following fields:
- `competition_id`: the ID of the competition in our dataset.
- `submission_path`: a `.csv` file with the predictions for the specified
  competition.

See more information by running `mlebench grade --help`.

You can also grade individual submissions using the `mlebench grade-sample` command. For example, to grade a submission for the Spaceship Titanic competition, you can run:

```console
mlebench grade-sample <PATH_TO_SUBMISSION> spaceship-titanic
```

See more information by running `mlebench grade-sample --help`.

```bash
mlebench grade-sample --data-dir  submission   --output-dir  competition_id
```
## Environment

We provide a base Docker image `mlebench-env` which is the base environment for our agents. This base image contains:
- Conda environment used to execute our agents. We optionally (default true) install Python packages in this environment which are commonly used across our agents. If you don't want to install these packages, set the `INSTALL_HEAVY_DEPENDENCIES` environment variable to `false` when building the image, by adding `--build-arg INSTALL_HEAVY_DEPENDENCIES=false` to the `docker build` command below
- Instructions for agents to follow when creating their submission
- Grading server for agents to use when checking that the structure of their submission is correct

Build this image by running:

```bash
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile . --build-arg INSTALL_HEAVY_DEPENDENCIES=false
```
## Agents

We purposefully designed our benchmark to not make any assumptions about the agent that produces submissions, so agents can more easily be evaluated on this benchmark. We evaluated three open-source agents; we discuss this procedure in [agents/README.md](agents/README.md).

## Extras

We include additional features in the MLE-bench repository that may be useful
for MLE-bench evaluation. These include a rule violation detector and
a plagiarism detector. We refer readers to
[extras/README.md](extras/README.md) for more information.

## Examples

We collect example usage of this library in the `examples/` directory, see [examples/README.md](examples/README.md) for more information.

## Experiments

We place the code specific to the experiments from our publication of the
benchmark in the `experiments/` directory:
- For instance, our competition splits are available in `experiments/splits/`.
- For a completed set of runs from a given agent, you can use the provided
`experiments/make_submission.py` script to compile its submission for grading.
- We release our methodology for the "familiarity" experiments in `experiments/familiarity/`, see [experiments/familiarity/README.md](experiments/familiarity/README.md) for more information.

## Dev

Note, when running `pytest` locally, be sure to accept the competition rules otherwise the tests will fail.

## Authors

Chan Jun Shern, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, Aleksander Mądry

## Citation

Please cite using the following BibTeX entry:
```
@article{chan2024mle-bench,
  title={MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering},
  author={Jun Shern Chan and Neil Chowdhury and Oliver Jaffe and James Aung and Dane Sherburn and Evan Mays and Giulio Starace and Kevin Liu and Leon Maksin and Tejal Patwardhan and Lilian Weng and Aleksander Mądry},
  year={2024},
  eprint={2410.07095},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.07095}
}
```


python3 -m vllm.entrypoints.openai.api_server \
    --model  "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2" \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda \
    --gpu-memory-utilization 0.9 \
    --max-model-len 13310 \
    --quantization gptq 


    # python run_agent.py --competition-set experiments/splits/chosen_competions.txt --agent-id aide --n-workers 5 --n-seeds 3
    AXERA-TECH/DeepSeek-R1-Distill-Qwen-7B-GPTQ-Int4
    "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"


uv pip install \
  --index-strategy unsafe-best-match  --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  -e .

# 1. Spooky author comp
# aide data_dir="data/aerial-cactus-identification" \
#       goal="Create a classifier capable of predicting whether a 32x32 aerial image contains a cactus" \
#       eval="Area Under the ROC Curve (AUC)" \
#       agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#       agent.steps=200 \
#       agent.code.max_new_tokens=2048 \
#       agent.code.temp=0.6 \
#       wandb.project="aerial-cactus-7b" \
#       exp_name="aerial-cactus_7b"


# 1. Spooky author comp
# aide data_dir="data/$competition_id/" \ 
#       goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" \
#       eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." \
#       agent.code.model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
#       agent.steps=200 \
#       agent.code.max_new_tokens=2048 \
#       agent.code.temp=0.6 \
#       wandb.project="spooky-author-14B" \
#       exp_name="14b_spooky_author"

 
# 2. aerial-cactus-identification
# aide data_dir="data/aerial-cactus-identification" \ 
#      goal="Create a classifier capable of predicting whether a 32x32 aerial image contains a cactus" \
#      eval="Area Under the ROC Curve (AUC)" \
#      agent.code.model= "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#      agent.steps=200 \
#      agent.code.max_new_tokens=2048 \
#      agent.code.temp=0.6 \
#      wandb.project="aerial-cactus-7b" \
#      exp_name="aerial-cactus_7b"

# 3. Random-Acts-0f-Pizza
# aide data_dir="data/$competition_id/" \ 
#      goal="Predict the probability that a textual request for pizza on Reddit resulted in the requester receiving pizza, probability that a textual request for pizza posted on Reddit will be successful" \
#      eval="Area Under the ROC Curve using the request text and associated metadata" \
#      agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
#      agent.steps=200 \
#      agent.code.temp=0.6 \
#      wandb.project="random-acts-of-pizza" \
#      exp_name="random-acts-of-pizza_DS_32B"


# 4. nomad2018-predict-transparent-conductors

# aide data_dir="data/nomad2018-predict-transparent-conductors/" \ 
#      goal="Predict formation energy (formation_energy_ev_natom) and bandgap energy (bandgap_energy_ev) for materials given their composition and structural properties" \
#      eval="Mean of column-wise Root Mean Squared Logarithmic Error (RMSLE) across the two target columns" \
#      agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#      agent.steps=200 \
#      agent.code.max_new_tokens=2048 \
#      agent.code.temp=0.6 \
#      wandb.project="aide-nomad2018" \
#      exp_name="7b_nomad2018"

## jigsaw-toxic-comment-classification-challenge

# aide data_dir="data/jigsaw-toxic-comment-classification-challenge"  \
#      goal="Predict the probability for each of six types of toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate) for each given comment text" \
#      eval="Mean column-wise ROC AUC" \
#      agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#      agent.steps=200 \
#      agent.code.max_new_tokens=2048 \
#      agent.code.temp=0.6 \
#      wandb.project="aide-jigsaw-comment" \
#      exp_name="7b_jigsaw"


# zip -r Deepseek-r1-32b_nomad2018-logs.zip ./logs/32b_nomad2018 &
# zip -r Deepseek-r1-32b_nomad2018-workspace.zip ./32b_nomad2018 &


# python run_agent.py --competition-set experiments/splits/chosen_competions.txt --agent-id aide 