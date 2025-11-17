We provide an example of collecting agent trajectories using Claude with reasoning as the agent and Mistral-7b-PRM as the verifier. We also experimented with two agent libraries: [Aviary](https://github.com/Future-House/aviary) and [Octotools](https://github.com/octotools/octotools). We recommend looking at those repositories for examples in collecting trajectories with those agents. `collect_math_example.py` recreates an example agent/verifier combination on the MATH dataset.

## Usage
To use this code, you can run the following command:

```bash
python collect_math.py --api_key_file [YOUR_ANTHROPIC_API_KEY_FILE] --model [YOUR_MODEL_CHOICE] --save_path [PATH_TO_SAVE_SCORES]
```