import anthropic
import os
import argparse
from datasets import load_dataset
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"

FEW_SHOT_EXAMPLE = r"""Here is an example of how you should respond:

Question:
The president can be any one of the 20 members, and the vice-president can be any one of the 19 remaining members. How many ways can you pick a president and a vice president?

Answer:
Step 1: There are 20 choices for president.
Step 2: There are 19 choices for vice president.
Step 3: Multiply to count ordered pairs: 20 × 19 = 380.
Step 4: The answer is: \boxed{380}

Now solve the following problem in the same style:
"""

SYSTEM_PROMPT = (
    "You are a meticulous math tutor. Solve each problem step by step, in contest style. "
    "Output each step on its own line numbered as Step 1:, Step 2:, etc. "
    "Finish with a final line: 'The answer is: \\boxed{<expression>}' (just the boxed expression). "
    "Do not include extra commentary or text outside this format.\n\n"
    + FEW_SHOT_EXAMPLE
)

def call_claude(client: anthropic.Anthropic, model: str, question: str,
                temperature: float = 0.0, max_tokens: int = 800) -> str:
    resp = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": question}],
    )
    blocks = getattr(resp, "content", []) or []
    txt_chunks = []
    for b in blocks:
        t = getattr(b, "text", None)
        if t:
            txt_chunks.append(t)
    return "\n".join(txt_chunks).strip()


def count_tokens(text: str) -> int:
    """Use Anthropic's tokenizer if available, else whitespace."""
    try:
        return anthropic.count_tokens(text)
    except Exception:
        return len(re.findall(r"\S+", text))


def _normalize_boxed_content(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.rstrip(" \t\r\n.")
    return s


def _balance_closing_braces(s: str) -> str:
    opens = 0
    for ch in s:
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens = max(opens - 1, 0)
    if opens > 0:
        s = s + ("}" * opens)
    return s


def extract_last_boxed_balanced(text: str):
    key = r"\boxed{"
    i = 0
    last_content = None
    n = len(text)
    while True:
        j = text.find(key, i)
        if j == -1:
            break
        # start of inner content
        k = j + len(key)
        depth = 1
        p = k
        while p < n and depth > 0:
            ch = text[p]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            p += 1
        if depth == 0:
            # matched at p-1
            content = text[k:p-1]
        else:
            # unbalanced; take until end and add missing braces
            content = text[k:]
            content = _balance_closing_braces(content)
        content = _normalize_boxed_content(content)
        last_content = content
        i = p  # continue searching after this block
    return last_content

def main():
    parser = argparse.ArgumentParser(description="Collect MATH trajectories.")
    parser.add_argument("--api_key_file", help="Path to anthropic key file")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022")
    parser.add_argument("--save_path", help="Path to save results")
    args = parser.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = open(args.api_key_file).read().strip()
    client = anthropic.Anthropic()
    model = args.model

    categories = [
        "algebra",
        "geometry",
        "counting_and_probability",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    ## only use the first category for this demo
    ds = load_dataset("EleutherAI/hendrycks_math", categories[0], split="test")
    ## only use the first five problems
    ds_subset = ds.select(range(5))
    txt_answers = []
    questions = []
    solved_list = []

    ## generate answer for each problem
    for i, ex in enumerate(ds_subset):
        q = ex["problem"]

        txt = call_claude(anthropic.Anthropic(), model=model, question=q)

        ## extract answers from generated text and true solution
        pred_boxed = extract_last_boxed_balanced(txt)
        pred_true = extract_last_boxed_balanced(ex["solution"])

        if pred_boxed == pred_true:
            solved = 1
        else:
            solved = 0
        solved_list.append(solved)
        txt_answers.append(txt)
        questions.append(q)

    rows = []
    results = []
    for j, s in enumerate(txt_answers):
        generated_tokens = []
        num_steps = []
        # extract only lines starting with "Step"
        steps = [ln.strip() for ln in s.splitlines() if ln.strip().startswith("Step")]
        # print(steps)
        # join with token
        transformed = ""
        for i, st in enumerate(steps):
            ## add special token to indicate step end
            transformed += st + (" ки\n" if i < len(steps) - 1 else " ки")
            # uq_problem_idx.append(f"{categories[0]}_{i}")
            generated_tokens.append(count_tokens(transformed))
            num_steps.append(i+1)
        results.append({"steps": transformed})

        solved_list_prob = [solved_list[j] for _ in range(len(num_steps))]
        uq_problem_idx = [f"{categories[0]}_{j}" for _ in range(len(num_steps))]

        # print(uq_problem_idx, generated_tokens, solved_list_prob, num_steps)
        for uq, tok, slv, ns in zip(uq_problem_idx, generated_tokens, solved_list_prob, num_steps):
            rows.append({
                "uq_problem_idx": uq,
                "generated_tokens": tok,
                "solved": slv,
                "num_steps": ns,
            })
    # print(rows)
    df = pd.DataFrame(rows)


    MODEL_NAME = "peiyi9979/math-shepherd-mistral-7b-prm"
    good_token = "+"
    bad_token  = "-"
    step_tag   = "ки"  

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto"  
    ).eval()

    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [good_id, bad_id]
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1]

    print(f"candidate_tokens = {candidate_tokens}  (good='{good_token}', bad='{bad_token}')")
    print(f"step_tag_id = {step_tag_id}  for tag '{step_tag}'")

    
    all_scores = []
    for i, result in enumerate(results):
        input_for_prm = f"{questions[i]} {result['steps']}"
        input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)  # no device ops

        with torch.no_grad():
            logits = model(input_id).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]  # prob of GOOD token

        step_scores = scores[input_id == step_tag_id]
  
        all_scores.append(step_scores.detach().cpu().tolist())

    # print(all_scores)
    scores_flattened = []
    for score in all_scores:
        scores_flattened.extend(score)
    # print(scores_flattened)

    df['judge_probability'] = scores_flattened
    print(df.head(20))

    if args.save_path:
        df.to_csv(args.save_path, index=False)

    
    

if __name__ == "__main__":
    main()