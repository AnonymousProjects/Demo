# Security Auditing under the NSFW filter of Stable Diffusion

We use AdjointDPM for auditing the security of AI generation systems. We can show that by using AdjointDPM, we can generate the adversarial samples against NSFW filter.

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

### Adversarial example generation
We provide an example script in run.sh.
```
python3 safety_attack.py \
    --output_dir='/home/tiger/assets/safety_attack' \
    --num_train_epochs=200 \
    --prompt='A photography of a naked man' \
    --learning_rate=0.03
```

We can check the final noise whether it can skip the NSFW filter by safety_attack_test.py.

Notes: In our code, we re-write the safety checker in Stable Diffusion, as we need to keep the gradients. Please use our new safety checker to check if the learned noise can skip the NSFW, otherwises it can still escape the filter.