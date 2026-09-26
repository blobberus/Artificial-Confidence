"""Model loading and the three readouts of the model's probability (protocol §1, §3, §4).

Every readout maps (question, four options) -> four probabilities summing to 1, in option order A-D.

    A  letter    softmax over the logits of the four answer-letter tokens after "Answer:"
    B  seq       softmax over the summed log-probability of each option's text
    C  seq_norm  softmax over the mean per-token log-probability of each option's text

All three condition on the same prompt (question plus the lettered options), so they differ only in
*where* the model's belief is read from (struggle S1).
"""
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {
    "stock":      "Qwen/Qwen3-8B",
    "uncensored": "huihui-ai/Huihui-Qwen3-8B-abliterated-v2",
}
LETTERS = ["A", "B", "C", "D"]
READOUTS = ["letter", "seq", "seq_norm"]

BNB = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load(variant):
    """Load one variant in 4-bit nf4. Returns (tok, model)."""
    model_id = MODELS[variant]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BNB, device_map="cuda:0")
    model.eval()
    return tok, model


def free_gpu():
    """Release GPU memory after the caller has dropped its model (`del model`),
    so the next variant loads into an empty card."""
    gc.collect()
    torch.cuda.empty_cache()


def build_prompt(tok, question, options):
    body = "\n".join(f"{L}. {t}" for L, t in zip(LETTERS, options))
    user = ("Answer the multiple-choice question with a single letter.\n\n"
            f"Question: {question}\n{body}")
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False,           # Qwen3: suppress the <think> block
    ) + "Answer:"


def letter_token_ids(tok):
    """Token ids of " A", " B", " C", " D" (the leading space matters)."""
    ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
    assert len(set(ids)) == 4, f"letter tokens collide: {tok.convert_ids_to_tokens(ids)}"
    return ids


@torch.no_grad()
def readout_letter(model, tok, question, options, letter_ids):
    """Readout A. The 4-way softmax renormalises away any mass on other tokens."""
    prompt = build_prompt(tok, question, options)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    last = model(**inputs).logits[0, -1].float()
    return torch.softmax(last[letter_ids], dim=-1).tolist()


@torch.no_grad()
def continuation_logprobs(model, tok, prompt, continuation):
    """Log-probability of each continuation token given the prompt.

    Prompt and continuation are tokenized separately and concatenated, so the boundary is exact:
    the logits at position prompt_len - 1 predict the first continuation token.
    """
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(continuation, add_special_tokens=False)
    ids = torch.tensor([p_ids + c_ids], device=model.device)
    logp = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
    targets = ids[0, len(p_ids):]
    return logp[len(p_ids) - 1:-1].gather(1, targets[:, None]).squeeze(1)


def option_logprobs(model, tok, question, options):
    """Per-token log-probabilities of each option's text (with a leading space) after the prompt."""
    prompt = build_prompt(tok, question, options)
    return [continuation_logprobs(model, tok, prompt, " " + text) for text in options]


def readout_sequence(option_lps, normalise):
    """Readout B (normalise=False, sum: favours short options) or C (normalise=True, mean: over-corrects)."""
    scores = torch.stack([lp.mean() if normalise else lp.sum() for lp in option_lps])
    return torch.softmax(scores, dim=-1).tolist()


def score_question(model, tok, question, options, letter_ids):
    """All three readouts for one question: {readout: [p_a, p_b, p_c, p_d]}."""
    lps = option_logprobs(model, tok, question, options)
    return {
        "letter":   readout_letter(model, tok, question, options, letter_ids),
        "seq":      readout_sequence(lps, normalise=False),
        "seq_norm": readout_sequence(lps, normalise=True),
    }
