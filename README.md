The is the repository documenting experiments for the paper: **"Optimizing Decomposition for Optimal Claim Verification"**.

Note: We have removed directories that might contain personal or sensitive information. Please specify your desired directory before reimplementing our work, places inclduing `src/utils/utils.py`, `src/utils/configs.py`, and `bash/*.sh`.

## Folder Structure
```shellscript
steps/   // callable scripts correspond to each step of experiments
src/     // source code of models, algorithms, data strcutures, metrics, etc. 
bash/    // bash scripts to submit experiment jobs
data/    // pre-processed data used in experiments
```

## Dataset Specification
The processed data can be found under `data/`. We prepare data in the following format. 
```json
{
    "text": "Sian Massey-Ellis is a professional football (soccer) assistant referee from England. She was born on October 5, 1985, in Coventry, England. Sian first began refereeing at the age of 16 and worked her way up through the ranks to become a Premier League assistant referee.",
    "topic": "Sian Massey-Ellis",
    "label": "NS",
    "subclaims": 
    [
        "Sian Massey-Ellis is a professional football (soccer) assistant refere from England.",
        "She was born on October 5, 1985, in Coventry, England.",
        "Sian first began refereeing at the age of 16 and worked her way up through the ranks to become a Premier League assistant referee."
    ]
}
```
Our data processing code can be found at `steps/prep`.

## Environment Setup
To run the code, you need to configure the environment by running the following command:
```bash
conda create -n dydecomp python=3.10
conda activate dydecomp
pip install -r requirements.txt
```

## Pilot Experiment
To replicate the pilot experiment results shown in Figure 3 and Figure 4, please run the following command:
```bash
qsub steps/pilot_experiments/pilot_experiments.sh
```
Please follow [FActScore](https://github.com/shmsw25/FActScore) repo to reconstruct *Inst-LLAMA-7B* used in our experiment. You can find our finetuned T5-3B on huggingface [ylu610/FT-T5-3B-Verifier](https://huggingface.co/ylu610/FT-T5-3B-Verifier).

To prepare the database (approximately 21G) for the retrieval verificaiton policy, you can either use the script from [FActScore](https://github.com/shmsw25/FActScore) repo or download it directly from the [Google Drive](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view).

## Experiment

### Step1: Deploy Decomposer
When using the decomposition LLM *Llama3-Inst-70B*, we first deploy it as a VLLM server using the following code.

```bash
qsub bash/start_server.sh
```
For the decomposition LLM *DeepSeek-V3*, this step is not required.

### Step2: Train Dynamic Decomposition Policy
```bash
qsub bash/train_dydecomp.sh
```
Note that the example shown in `bash/train_dydecomp` uses decomposition LLM *Llama3-Inst-70B* and verifier *Llama3-Inst-8B* with the *retrieval* verification policy. To find other options, please refer to the help statements of arguments `decomposer_name`, `verifier_name`, and `verify_policy` in `src/utils/arguments.py`.

### Step3: Evaluate Dynamic Decomposition Policy
Please run the following command to evaluate the trained decomposition policy on different datasets and atomicities.
```bash
qsub bash/eval_dydecomp.sh
```

## Citation
If you use this code, please cite the following paper:
```bibtex

```