# WildChat-50m

This repository contains all code, results and other artifacts from the paper introducing the WildChat-50m dataset and the Re-Wild model family.

## Links

[Our Dataset](https://huggingface.co/collections/nyu-dice-lab/wildchat-50m-679a5df2c5967db8ab341ab7)

[Our Models, Including Re-Wild](https://huggingface.co/collections/nyu-dice-lab/wildchat-50m-models-679a5bef432ea93dba6d03b1)

[Our Judgment Datasets](https://huggingface.co/collections/nyu-dice-lab/wildchat-50m-judgments-679a63f5b867072a3339b8ac)

[Extended Evalchemy Results](https://huggingface.co/datasets/nyu-dice-lab/wildchat-50m-extended-results)

### Weights and Biases Logs

These will be made available with a later release.

## Generating Model Responses

The dataset was produced using `generate_model_responses.py`. Although we focused on the WildChat-1M dataset, we believe that the code should generalize reasonably easily to other HuggingFace datasets which contain a column of conversations.

### Viewing Sample Model Responses

You can find sample conversations, in custom HTML format, as well as judgments, in the `model_responses` directory.

## Configs

The `configs` directory contains samples of the scripts used to launch our training runs, which were conducted using [axolotl](https://github.com/axolotl-ai-cloud/axolotl) for SFT and [open-instruct](https://github.com/allenai/open-instruct) for DPO.

## Notebooks

Our `plotting` notebook reproduces the plots associated with this paper. The `conversation_processing` notebook generates the custom-formatted HTML conversations for pairs of models side-by-side, which we utilize in the appendix of our paper. The `mt_bench_jsonl_to_html` generates the custom-formatted HTML conversations for single models with judgments. `textual_similarity` was used to compute the similarity scores between models.

## Licensing

All code and data authored by us is released under the Apache 2.0 License. All data not authored by us is subject to its original license(s).

## Citation

If you find our work useful, please consider citing us!

```
@misc{feuer2025wildchat50mdeepdiverole,
      title={WILDCHAT-50M: A Deep Dive Into the Role of Synthetic Data in Post-Training}, 
      author={Benjamin Feuer and Chinmay Hegde},
      year={2025},
      eprint={2501.18511},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.18511}, 
}
```
