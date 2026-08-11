import accelerate
from accelerate import Accelerator,InitProcessGroupKwargs
import argparse
from collections import defaultdict
import datasets
import datetime
from huggingface_hub import login
import json
import logging
import math
import numpy as np
import os
from termcolor import colored
import traceback
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    GPT2Tokenizer,
    OPTForCausalLM
)
import torch
import torch.nn.functional as F  
from tqdm import tqdm


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, torch.finfo(logits.dtype).min)
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return (2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value)


def filter_logits_top_p(logits, top_p, negative_multiplier=False):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    if negative_multiplier:
        filtered_logits = logits.masked_fill(valid_indices == 0, 1000)
    else:
        filtered_logits = logits.masked_fill(valid_indices == 0, -1000)
    return filtered_logits


SOURCE_START_MARKERS = (
    "article:",
    "document:",
    "passage:",
)

SOURCE_END_MARKERS = (
    "summarize the article",
    "summarize the document",
    "summarize the passage",
    "give a one-sentence summaries of the documents",
    "using only the references listed above",
    "tl;dr:",
    "summary:",
    "question:",
)


def find_subsequence(haystack, needle):
    if not needle:
        return 0
    last_start = len(haystack) - len(needle)
    for start_idx in range(last_start + 1):
        if haystack[start_idx : start_idx + len(needle)] == needle:
            return start_idx
    return -1


def longest_common_suffix_length(text_a, text_b):
    max_len = min(len(text_a), len(text_b))
    count = 0
    while count < max_len and text_a[-1 - count] == text_b[-1 - count]:
        count += 1
    return count


def trim_span_boundaries(context_string, start_idx, end_idx):
    while start_idx < end_idx and context_string[start_idx].isspace():
        start_idx += 1
    while end_idx > start_idx and context_string[end_idx - 1].isspace():
        end_idx -= 1
    if start_idx >= end_idx:
        return None
    return start_idx, end_idx


def extract_source_span(context_string, without_context_string=None):
    lowered_context = context_string.lower()

    start_idx = None
    for marker in SOURCE_START_MARKERS:
        marker_idx = lowered_context.find(marker)
        if marker_idx != -1:
            start_idx = marker_idx + len(marker)
            break

    search_start = start_idx if start_idx is not None else 0
    end_idx = len(context_string)
    found_end = False
    for marker in SOURCE_END_MARKERS:
        marker_idx = lowered_context.find(marker, search_start)
        if marker_idx != -1 and marker_idx < end_idx:
            end_idx = marker_idx
            found_end = True

    if found_end:
        span = trim_span_boundaries(context_string, search_start, end_idx)
        if span is not None:
            return span

    if without_context_string:
        suffix_start = context_string.rfind(without_context_string)
        if suffix_start > 0:
            span = trim_span_boundaries(context_string, search_start, suffix_start)
            if span is not None:
                return span

        common_suffix_len = longest_common_suffix_length(context_string, without_context_string)
        if common_suffix_len >= 16:
            span = trim_span_boundaries(
                context_string,
                search_start,
                len(context_string) - common_suffix_len,
            )
            if span is not None:
                return span

    return None


def resolve_boost_scope_positions(
    tokenizer,
    context_string,
    context_input_ids,
    boost_source_scope,
    without_context_string=None,
):
    full_positions = list(range(context_input_ids.size(1)))
    if boost_source_scope == "full_context":
        return full_positions

    if boost_source_scope not in {"article_only", "source_only"}:
        raise ValueError(f"Unsupported boost source scope: {boost_source_scope}")

    source_span = extract_source_span(context_string, without_context_string=without_context_string)
    if source_span is None:
        logger.warning("Failed to locate source span in prompt. Falling back to full-context boosting.")
        return full_positions

    source_start, source_end = source_span
    prefix_ids = tokenizer.encode(context_string[:source_start], add_special_tokens=False)
    prefix_plus_source_ids = tokenizer.encode(context_string[:source_end], add_special_tokens=False)
    full_prompt_ids = tokenizer.encode(context_string, add_special_tokens=False)
    full_context_ids = context_input_ids[0].tolist()

    prompt_start = find_subsequence(full_context_ids, full_prompt_ids)
    if prompt_start == -1:
        logger.warning("Failed to align prompt tokens for source-only scope. Falling back to full-context boosting.")
        return full_positions

    scope_positions = list(
        range(
            prompt_start + len(prefix_ids),
            prompt_start + len(prefix_plus_source_ids),
        )
    )
    if not scope_positions:
        logger.warning("Source-only scope resolved to an empty span. Falling back to full-context boosting.")
        return full_positions

    return scope_positions


def get_context_token_ids(context_input_ids, position_indices=None):
    if position_indices is None:
        return torch.unique(context_input_ids)
    return torch.unique(context_input_ids[0, position_indices])

def create_boost_mask(logits, context_tokens, delta):
    boost_mask = torch.zeros_like(logits)
    for token in context_tokens:
        boost_mask[..., token] = delta
    return boost_mask


def parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean flag value: {value}")


def get_context_token_positions(context_input_ids, position_indices=None):
    token_to_positions = defaultdict(list)
    ordered_unique_tokens = []
    seen = set()

    if position_indices is None:
        position_indices = list(range(context_input_ids.size(1)))

    for local_pos, absolute_pos in enumerate(position_indices):
        token = int(context_input_ids[0, absolute_pos].item())
        token_to_positions[token].append(local_pos)
        if token not in seen:
            ordered_unique_tokens.append(token)
            seen.add(token)

    return ordered_unique_tokens, token_to_positions



def compute_jsd(logits_with_context, logits_without_context):
    """calculate Jensen-Shannon divergence with numerical stability"""
    p = F.softmax(logits_with_context, dim=-1)
    q = F.softmax(logits_without_context, dim=-1)
    
    p = torch.clamp(p, min=1e-10, max=1.0)
    q = torch.clamp(q, min=1e-10, max=1.0)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    m = 0.5 * (p + q)
    m = torch.clamp(m, min=1e-10, max=1.0)
    m = m / m.sum(dim=-1, keepdim=True)
    
    kl_p_m = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_q_m = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    return jsd.item()


def get_adaptive_delta(logits_with_context, logits_without_context, args):
    """get global adaptive boost delta with safety checks"""
    try:
        jsd = compute_jsd(logits_with_context, logits_without_context)
        
        if torch.isnan(torch.tensor(jsd)) or jsd < 0:
            return args.min_delta
        
        jsd = min(max(jsd, 0.0), 1.0)
        
        delta = args.min_delta + (args.max_delta - args.min_delta) * jsd
        
        return delta
    
    except Exception as e:
        logger.warning(f"Error in computing adaptive delta: {e}, using default delta")
        return args.min_delta
    
def calculate_semantic_similarity(model, token_id, context_embeddings):
    """calculate semantic similarity"""
    with torch.no_grad():
        # get token embedding
        token_embedding = model.get_input_embeddings()(
            torch.tensor([token_id]).to(context_embeddings.device)
        )
        # cosine similarity
        similarities = F.cosine_similarity(
            token_embedding,
            context_embeddings,
            dim=1
        )
        
        return similarities.mean().item()
    
def get_attention_scores(model, input_ids, position_indices=None):
    """get attention scores"""
    if position_indices is None:
        position_indices = list(range(input_ids.size(1)))
    try:
        # use output_attentions parameters
        outputs = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_attentions=True,
        )
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            last_layer_attention = outputs.attentions[-1]  
            averaged_attention = last_layer_attention.mean(dim=1)  
            scores = averaged_attention[0, -1, :input_ids.size(1)]
            scores = scores[position_indices]
            return F.softmax(scores, dim=-1)
            
    except Exception as e:
        print(f"Warning: Failed to get attention scores: {e}")
        
    try:
        # or use hidden states
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  
        query = hidden_states[:, -1:]  
        key = hidden_states[:, position_indices]  
        attention = torch.matmul(query, key.transpose(-2, -1))  
        attention = attention / math.sqrt(query.size(-1))
        scores = F.softmax(attention, dim=-1)[0, 0]  
        return scores
        
    except Exception as e:
        print(f"Warning: Failed to compute attention using hidden states: {e}")
    
    # Fallback
    return torch.ones(len(position_indices)).to(input_ids.device) / len(position_indices)


def get_effective_local_weights(args):
    attn_weight = 0.0 if args.disable_attention_score else max(args.lambda1, 0.0)
    semantic_weight = 0.0 if args.disable_semantic_score else max(args.lambda2, 0.0)

    if args.renormalize_component_weights:
        total = attn_weight + semantic_weight
        if total > 0:
            attn_weight /= total
            semantic_weight /= total

    return attn_weight, semantic_weight


def normalize_token_importance(importance_by_token, mode):
    if mode == "none" or not importance_by_token:
        return importance_by_token

    values = list(importance_by_token.values())
    if mode == "mean1":
        denom = sum(values) / len(values)
    elif mode == "sum1":
        denom = sum(values)
    else:
        raise ValueError(f"Unsupported importance normalization mode: {mode}")

    if denom <= 0:
        return importance_by_token

    return {token: value / denom for token, value in importance_by_token.items()}


def build_occurrence_aggregated_boost_mask(
    logits,
    ordered_unique_tokens,
    token_to_positions,
    attention_scores,
    semantic_score_by_token,
    args,
):
    attn_weight, semantic_weight = get_effective_local_weights(args)

    if attn_weight == 0.0 and semantic_weight == 0.0:
        return create_boost_mask(logits, ordered_unique_tokens, args.adaptive_delta)

    importance_by_token = {}
    for token in ordered_unique_tokens:
        positions = token_to_positions[token]

        if attn_weight > 0.0:
            if attention_scores is None:
                raise ValueError("Attention scores are required when attention weighting is enabled.")
            position_scores = attention_scores[positions]
            if args.token_attention_aggregation == "sum":
                attn_score = position_scores.sum().item()
            elif args.token_attention_aggregation == "mean":
                attn_score = position_scores.mean().item()
            elif args.token_attention_aggregation == "max":
                attn_score = position_scores.max().item()
            else:
                raise ValueError(
                    f"Unsupported token attention aggregation: {args.token_attention_aggregation}"
                )
        else:
            attn_score = 0.0

        semantic_score = semantic_score_by_token.get(token, 0.0) if semantic_weight > 0.0 else 0.0
        importance_by_token[token] = attn_weight * attn_score + semantic_weight * semantic_score

    importance_by_token = normalize_token_importance(
        importance_by_token,
        args.token_importance_normalization,
    )

    boost_mask = torch.zeros_like(logits)
    for token, importance in importance_by_token.items():
        boost_mask[..., token] = args.adaptive_delta * importance
    return boost_mask

def decode(args, batch_input_ids, dec_depth, model, tokenizer):
    batch_size = args.per_device_eval_batch_size
    assert batch_input_ids.size(1) == args.context_size
    assert args.decode_truncate_len >= 0
    assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
    use_global_mode = parse_bool_flag(args.use_global)

    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        scope_position_indices = resolve_boost_scope_positions(
            tokenizer,
            args.current_context_string,
            unit_context_input_ids,
            args.boost_source_scope,
            without_context_string=getattr(args, "current_without_context_string", None),
        )
        context_tokens = get_context_token_ids(unit_context_input_ids, scope_position_indices)
        ordered_unique_tokens, token_to_positions = get_context_token_positions(
            unit_context_input_ids,
            scope_position_indices,
        )
    else:
        raise ValueError("context cannot be none")
    
    history_decode_ids = None
    past_key_values = None

    # calculate context embeddings for semantic similarity
    with torch.no_grad():
        context_embeddings = model.get_input_embeddings()(unit_context_input_ids[0, scope_position_indices])

    semantic_score_by_token = {}
    if not use_global_mode and not args.disable_semantic_score:
        for token in ordered_unique_tokens:
            semantic_sim = calculate_semantic_similarity(
                model,
                token,
                context_embeddings
            )
            semantic_score_by_token[token] = (semantic_sim + 1) / 2

    if args.model_category == 'seq2seq':
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            batch_input_ids[:, :args.context_size].clone(), dict(), None
        )
        history_decode_ids = model._prepare_decoder_input_ids_for_generation(
            batch_input_ids.size(0),
            model_kwargs=model_kwargs,
            device=batch_input_ids.device,
        )
    else:
        model_kwargs = None

    for _i in range(dec_depth):
        if args.model_category == 'causal':
            if 'llama' in args.model_name_or_path.lower() or 'mistral' in args.model_name_or_path.lower():
                if past_key_values is not None:
                    model_inputs = {
                        "input_ids": unit_context_input_ids[:, -1:],
                        "attention_mask": torch.ones_like(unit_context_input_ids),
                        "past_key_values": past_key_values,
                        "use_cache": True
                    }
                else:
                    model_inputs = {
                        "input_ids": unit_context_input_ids,
                        "attention_mask": torch.ones_like(unit_context_input_ids),
                        "use_cache": True
                    }
            else:
                model_inputs = model.prepare_inputs_for_generation(
                    unit_context_input_ids, 
                    past_key_values=past_key_values
                )
        elif args.model_category == 'seq2seq':
            model_inputs = model.prepare_inputs_for_generation(
                history_decode_ids, 
                **model_kwargs
            )
        else:
            raise ValueError("model category not supported")

        outputs = model(**model_inputs, output_hidden_states=False, output_attentions=True)

        # get logits
        logits = outputs.logits[:, -1:, :].clone().contiguous()
        # apply global adaptive boost first
        boost_mask = create_boost_mask(logits, context_tokens, args.adaptive_delta)

        # token-wise adaptive boost base on importance 
        if not use_global_mode:
            if args.tokenwise_implementation == "legacy":
                attn_weight, semantic_weight = get_effective_local_weights(args)
                if attn_weight == 0.0 and semantic_weight == 0.0:
                    boost_mask = create_boost_mask(logits, context_tokens, args.adaptive_delta)
                else:
                    attention_scores = None
                    if attn_weight > 0.0:
                        attention_scores = get_attention_scores(
                            model,
                            unit_context_input_ids,
                            scope_position_indices,
                        )
                    for token_idx, token in enumerate(context_tokens):
                        attn_score = (
                            attention_scores[token_idx].item()
                            if attn_weight > 0.0 and attention_scores is not None and token_idx < attention_scores.size(0)
                            else 0.0
                        )
                        semantic_score = (
                            semantic_score_by_token.get(int(token), 0.0)
                            if semantic_weight > 0.0
                            else 0.0
                        )
                        importance = attn_weight * attn_score + semantic_weight * semantic_score
                        boost_mask[..., token] = args.adaptive_delta * importance
            elif args.tokenwise_implementation == "occurrence_agg":
                attention_scores = (
                    get_attention_scores(
                        model,
                        unit_context_input_ids,
                        scope_position_indices,
                    )
                    if not args.disable_attention_score else None
                )
                boost_mask = build_occurrence_aggregated_boost_mask(
                    logits,
                    ordered_unique_tokens,
                    token_to_positions,
                    attention_scores,
                    semantic_score_by_token,
                    args,
                )
            else:
                raise ValueError(
                    f"Unsupported token-wise implementation: {args.tokenwise_implementation}"
                )


        enhanced_logits = logits + boost_mask

        score = filter_logits_top_p(enhanced_logits, top_p=args.filter_top_p)
        projected_logits = logits_sampling_projection(score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
        
        simplex = torch.nn.functional.softmax(projected_logits, dim=-1)
        real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)

        if args.model_category == 'causal':
            unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)

        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

        if args.model_category == 'causal':
            past_key_values = outputs.past_key_values
        elif args.model_category == 'seq2seq':
            model_kwargs["past_key_values"] = outputs.past_key_values

        if real_token_ids_list[0][-1] == model.generation_config.eos_token_id:
            break

    # output
    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
    else:
        init_context_input_ids = None
        context_sequences = None

    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'), skip_special_tokens=True)
    logger.info(f"context: {context_sequences}")
    logger.info(f"sampled: {colored(str(sampled_sequences), 'red')}")

    return history_decode_ids, init_context_input_ids, None, sampled_sequences, context_sequences, None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument("--init_blank_language_model", action="store_true", help="Whether or not to use a completely blank LM.")
    parser.add_argument(
        "--file_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--train_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--decode_truncate_len", type=int, default=50, help="",
    ) 
    parser.add_argument(
        "--decode_depth", type=int, default=2, help="",
    )
    parser.add_argument(
        "--projection_top_p", type=float, default=0.2, help="",
    )
    parser.add_argument(
        "--filter_top_p", type=float, default=1.0, help="",
    )
    parser.add_argument(
        "--filter_top_p_prior", type=float, default=1.0, help="",
    )
    parser.add_argument("--big_model_inference", type=str, default="no")
    parser.add_argument(
        "--force_model_name_or_path",
        type=str,
        default=None,
        help="Optional model path/name override. If set, bypasses get_small_model_name remap."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/huggingface",
        help="Hugging Face cache directory."
    )
    parser.add_argument(
        "--boost_source_scope",
        type=str,
        default="full_context",
        choices=("full_context", "article_only", "source_only"),
        help="Which part of the prompt contributes eligible boost tokens."
    )

    # add adaptive related parameters
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1.0,
        help="Minimum boost value for adaptive delta"
    )
    parser.add_argument(
        "--max_delta",
        type=float,
        default=3.0,
        help="Maximum boost value for adaptive delta"
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.6,
        help="Weight for attention score in local mode"
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.4,
        help="Weight for semantic score in local mode"
    )
    parser.add_argument(
        "--use_global",
        type=str,
        default="false",
        help="Whether to use only global boost (true) or add local boost (false)"
    )
    parser.add_argument(
        "--disable_distribution_jsd",
        action="store_true",
        help="Disable the global JSD-based delta adaptation and use a fixed delta instead."
    )
    parser.add_argument(
        "--fixed_delta",
        type=float,
        default=None,
        help="Optional fixed delta used when JSD adaptation is disabled."
    )
    parser.add_argument(
        "--disable_attention_score",
        action="store_true",
        help="Disable the attention-score component in token-wise mode."
    )
    parser.add_argument(
        "--disable_semantic_score",
        action="store_true",
        help="Disable the semantic-similarity component in token-wise mode."
    )
    parser.add_argument(
        "--renormalize_component_weights",
        action="store_true",
        help="Renormalize the active token-wise component weights to sum to 1."
    )
    parser.add_argument(
        "--tokenwise_implementation",
        type=str,
        default="legacy",
        choices=("legacy", "occurrence_agg"),
        help="Legacy unique-token token-wise scoring or occurrence-aware aggregated scoring."
    )
    parser.add_argument(
        "--token_attention_aggregation",
        type=str,
        default="sum",
        choices=("sum", "mean", "max"),
        help="How to aggregate positional attention scores for repeated context tokens."
    )
    parser.add_argument(
        "--token_importance_normalization",
        type=str,
        default="none",
        choices=("none", "mean1", "sum1"),
        help="Optional normalization applied to token-wise importance values."
    )

    args = parser.parse_args()

    return args


def get_small_model_name(original_model_name):
    """map large models to smaller ones for test"""
    model_mapping = {  
        "huggyllama/llama-7b": "meta-llama/Meta-Llama-3-8B-Instruct",  
    }
    return model_mapping.get(original_model_name, "meta-llama/Meta-Llama-3-8B-Instruct")  ##mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Llama-2-13b-chat-hf



def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    hf_token = os.environ.get("HF_TOKEN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # input
    args.file_mode = args.file_mode.split('|')
    assert args.file_mode[0] == "fin"
    assert os.path.exists(args.file_mode[1])
    fin_path = args.file_mode[1]
    fin_data_pairs = []
    with open(fin_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines)-1):  
            data = json.loads(lines[i])
            next_data = json.loads(lines[i+1])      
            # paris of with/without context input
            if (data['input_index'] == next_data['input_index'] and 
                data.get('assigned_process') == 0 and 
                next_data.get('assigned_process') == 1):
                fin_data_pairs.append({
                    'with_context': data,
                    'without_context': next_data
                })
    # model name
    first_data = fin_data_pairs[0]['with_context'] 
    original_model_name = first_data['assigned_model'].split('|')[0]
    if args.force_model_name_or_path:
        args.model_name_or_path = args.force_model_name_or_path
        logger.info("Using forced model override via --force_model_name_or_path")
    else:
        args.model_name_or_path = get_small_model_name(original_model_name)
    logger.info(f"Original model: {original_model_name}")
    logger.info(f"Using model: {args.model_name_or_path}")

    try:
        loaded_with_device_map = False
        # tokenizer
        logger.info("Loading tokenizer...")
        if 'llama' in args.model_name_or_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                use_fast=True,
                cache_dir=cache_dir,
                token=hf_token,
            )
        elif 'opt' in args.model_name_or_path.lower():
            tokenizer = GPT2Tokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=cache_dir,
                token=hf_token,
            )
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=cache_dir,
                use_fast=not args.use_slow_tokenizer,
                token=hf_token,
            )
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")

        # model
        logger.info("Loading model...")
        if 'llama' in args.model_name_or_path.lower():
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                device_map='auto',
                cache_dir=cache_dir,
                token=hf_token,
            )
            loaded_with_device_map = True
        elif 'opt' in args.model_name_or_path.lower():
            model = OPTForCausalLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                token=hf_token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                token=hf_token,
            )

        if not loaded_with_device_map:
            model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully and moved to GPU")
        args.is_llama = 'llama' in args.model_name_or_path.lower()
        args.model_category = 'causal'
        args.vocab_size = model.config.vocab_size
        args.hidden_size = model.config.hidden_size
        args.one_hot_value = 5.0
        args.tokenizer = tokenizer

        # generate
        export_list = []
        args.orig_decode_truncate_len = args.decode_truncate_len

        with torch.no_grad():
            for data_pair in tqdm(fin_data_pairs, desc="Processing inputs"):
                try:
                    with_context_data = data_pair['with_context']
                    without_context_data = data_pair['without_context']
                    
                    args.assigned_weight = with_context_data.get('assigned_weight', 1.0)
                    args.filter_top_p = with_context_data.get('filter_p', getattr(args, 'filter_top_p', 1.0))
                    args.filter_top_p_prior = with_context_data.get('filter_p_prior', getattr(args, 'filter_top_p_prior', 1.0))
                    
                    # global adaptive delta
                    with_context_input = tokenizer(
                        with_context_data['context_string'], 
                        return_tensors="pt"
                    ).input_ids.to(device)
                    
                    without_context_input = tokenizer(
                        without_context_data['context_string'],
                        return_tensors="pt"
                    ).input_ids.to(device)
                    
                    with_context_outputs = model(with_context_input)
                    without_context_outputs = model(without_context_input)
                    
                    with_context_logits = with_context_outputs.logits[:,-1,:]
                    without_context_logits = without_context_outputs.logits[:,-1,:]
                    
                    if args.disable_distribution_jsd:
                        if args.fixed_delta is not None:
                            args.adaptive_delta = args.fixed_delta
                        else:
                            args.adaptive_delta = args.min_delta
                    else:
                        args.adaptive_delta = get_adaptive_delta(with_context_logits, without_context_logits, args)
                    logger.info(f"Sample {with_context_data['input_index']}")
                    
                    input_ids = torch.tensor(
                        tokenizer.encode(with_context_data['context_string'], add_special_tokens=True),
                        dtype=torch.long
                    ).unsqueeze(0).to(device)
                    args.current_context_string = with_context_data['context_string']
                    args.current_without_context_string = without_context_data['context_string']
                    
                    args.context_size = input_ids.size(1)
                    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
                    
                    if args.decode_truncate_len < 0:
                        logger.warning(f"Skipping long input {with_context_data['input_index']}")
                        continue
                    
                    # decode
                    with torch.cuda.amp.autocast():
                        history_decode_ids, init_context_ids, _, sampled_sequences, context_sequences, _ = \
                            decode(args, input_ids, args.decode_depth, model, tokenizer)

                    # results
                    export_dict = {
                        'tokens': [history_decode_ids.tolist()[0]],
                        'string': [sampled_sequences[0]],
                        'input_index': with_context_data['input_index'],
                        'output_index': len(export_list),
                        'assigned_model': args.model_name_or_path,
                        'original_model': original_model_name,
                        'assigned_weight': with_context_data.get('assigned_weight', 1.0),
                        'assigned_process': 0,
                        'adaptive_delta': args.adaptive_delta,  
                        'min_delta': args.min_delta,  
                        'max_delta': args.max_delta,  
                        'use_global': args.use_global,
                        'lambda1': args.lambda1,
                        'lambda2': args.lambda2,
                        'disable_distribution_jsd': args.disable_distribution_jsd,
                        'fixed_delta': args.fixed_delta,
                        'disable_attention_score': args.disable_attention_score,
                        'disable_semantic_score': args.disable_semantic_score,
                        'tokenwise_implementation': args.tokenwise_implementation,
                        'token_attention_aggregation': args.token_attention_aggregation,
                        'token_importance_normalization': args.token_importance_normalization,
                        'renormalize_component_weights': args.renormalize_component_weights,
                        'boost_source_scope': args.boost_source_scope,
                        'context': context_sequences[0] if context_sequences else None,
                    }
                    export_list.append(export_dict)
                    mode_str = "Global" if parse_bool_flag(args.use_global) else "Token-wise"
                    logger.info(f"Processed input {with_context_data['input_index']} using {mode_str} boosting mode")
                    logger.info(f"Global adaptive delta: {args.adaptive_delta}")

                except Exception as e:
                    logger.error(f"Error processing input {with_context_data['input_index']}: {str(e)}")
                    logger.error("Error details:", exc_info=True)
                    continue

        # output results
        output_dir = args.output_dir or "./output/re_llama3_8b"
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.basename(fin_path)

        suffix_parts = [
            f"delta{args.min_delta}-{args.max_delta}",
            f"l1{args.lambda1}",
            f"l2{args.lambda2}",
            f"global{args.use_global}",
        ]
        if args.disable_distribution_jsd:
            suffix_parts.append("nojsd")
        if args.fixed_delta is not None:
            suffix_parts.append(f"fixeddelta{args.fixed_delta}")
        if not parse_bool_flag(args.use_global):
            suffix_parts.append(f"impl{args.tokenwise_implementation}")
            if args.disable_attention_score:
                suffix_parts.append("noattn")
            if args.disable_semantic_score:
                suffix_parts.append("nosem")
            if args.renormalize_component_weights:
                suffix_parts.append("renormw")
            suffix_parts.append(f"agg{args.token_attention_aggregation}")
            suffix_parts.append(f"norm{args.token_importance_normalization}")
        if args.boost_source_scope != "full_context":
            suffix_parts.append(f"scope{args.boost_source_scope}")

        out_json_fn = (
            f"{base_filename}.output_topp{args.projection_top_p}_"
            f"genlen{args.decode_depth}_"
            + "_".join(suffix_parts)
            + ".jsonl"
        )

        out_json_fn = os.path.join(output_dir, out_json_fn)
        os.makedirs(os.path.dirname(out_json_fn), exist_ok=True)

        with open(out_json_fn, mode="w") as f_out:
            for export in export_list:
                f_out.write(json.dumps(export))
                f_out.write("\n")

        logger.info(f"Successfully processed {len(export_list)} out of {len(fin_data_pairs)} inputs")
        logger.info(f"Results saved to {out_json_fn}")
        logger.info(f"Processing summary:")
        logger.info(f"- Delta range: {args.min_delta} - {args.max_delta}")
        logger.info(f"- Global mode: {args.use_global}")
        logger.info(f"- Disable JSD: {args.disable_distribution_jsd}")
        logger.info(f"- Fixed delta: {args.fixed_delta}")
        logger.info(f"- Lambda1 (attention): {args.lambda1}")
        logger.info(f"- Lambda2 (semantic): {args.lambda2}")
        logger.info(f"- Disable attention: {args.disable_attention_score}")
        logger.info(f"- Disable semantic: {args.disable_semantic_score}")
        logger.info(f"- Token-wise implementation: {args.tokenwise_implementation}")
        logger.info(f"- Attention aggregation: {args.token_attention_aggregation}")
        logger.info(f"- Importance normalization: {args.token_importance_normalization}")
        logger.info(f"- Renormalize component weights: {args.renormalize_component_weights}")
        logger.info(f"- Boost source scope: {args.boost_source_scope}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error("Error details:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
