import collections
import math
import re
import string

import numpy as np
# from pytorch_lightning.utilities import print
from tqdm import tqdm
from transformers import BasicTokenizer


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def lcs_match(tokenizer, context_text, tokenized_text, max_dist):
    """Longest-common-substring algorithm. From https://github.com/zihangdai/xlnet
    This will be used with sentencepiece tokenizers.
    """
    n, m = len(context_text), len(tokenized_text)
    max_n, max_m = 1024, 1024
    if n > max_n or m > max_m:
        max_n = max(n, max_n)
        max_m = max(m, max_m)
    f = np.zeros((max_n, max_m), dtype=np.float32)
    g = {}

    ### longest common sub sequence
    # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
    for i in range(n):
        # note(zhiliny):
        # unlike standard LCS, this is specifically optimized for the setting
        # because the mismatch between sentence pieces and original text will
        # be small
        for j in range(i - max_dist, i + max_dist):
            if j >= m or j < 0:
                continue

            if i > 0:
                g[(i, j)] = 0
                f[i, j] = f[i - 1, j]

            if j > 0 and f[i, j - 1] > f[i, j]:
                g[(i, j)] = 1
                f[i, j] = f[i, j - 1]

            f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
            if tokenizer.preprocess_text(context_text[i]) == tokenized_text[j] and f_prev + 1 > f[i, j]:
                g[(i, j)] = 2
                f[i, j] = f_prev + 1
    return f, g


def convert_index(index, pos, M=None, is_start=True):
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


#################################################################################
# Metrics utilities                                                             #
# Most of the evaluation utilities are from: https://github.com/zihangdai/xlnet #
#################################################################################
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores_for_a_question(qas_id, gold_answers, preds):
    if not gold_answers:
        # For unanswerable questions, only correct answer is empty string
        gold_answers = [""]

    if qas_id not in preds:
        print("Missing prediction for %s" % qas_id)

    prediction = preds[qas_id]
    exact_scores = max(compute_exact(a, prediction) for a in gold_answers)
    f1_scores = max(compute_f1(a, prediction) for a in gold_answers)
    return exact_scores, f1_scores


def get_raw_scores(examples, preds, qa_mode):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    for example in examples:
        if qa_mode == 'FastQA':
            for i in range(len(example.q_ids)):
                qas_id = example.q_ids[i]
                gold_answers = [normalize_answer(example.answers[i][j]["text"]) for j in range(len(example.answers[i]))]
                exact_scores[qas_id], f1_scores[qas_id] = get_raw_scores_for_a_question(qas_id, gold_answers, preds)
        else:
            qas_id = example.uid
            gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]
            exact_scores[qas_id], f1_scores[qas_id] = get_raw_scores_for_a_question(qas_id, gold_answers, preds)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        # Some examples might not get predictions, only average on examples that do.
        valid_qid_list = [qid for qid in qid_list if qid in exact_scores]
        total = len(valid_qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in valid_qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in valid_qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return (
        100.0 * best_score / len(scores),
        best_thresh,
        1.0 * has_ans_score / has_ans_cnt,
    )


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            print(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text,
                tok_ns_text,
            )
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def normalqa_compute_predictions_logits(  # noqa: C901
        all_examples,
        all_features,
        all_results,
        tokenizer,
        max_ans_len,
        nbest_size,
        null_score_diff_threshold,
        version_2_with_negative,
        tokenizer_type="wordpiece",
        verbose_logging=False,
):
    example_id_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_id_to_features[feature.example_uid].append(feature)

    id_to_result = {result.uid: result for result in all_results}

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example in tqdm(all_examples, desc="Compute predictions from logits"):
        features = example_id_to_features[example.uid]

        # Some examples might fail to parse into features. There's no predictions for these examples.
        if len(features) == 0:
            continue

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = id_to_result[feature.uid]
            start_indexes = _get_best_indexes(result.start_logits, nbest_size)
            end_indexes = _get_best_indexes(result.end_logits, nbest_size)

            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if not (start_index in feature.token_to_orig_map or start_index in feature.tok_start_to_orig_index):
                        continue
                    if not (end_index in feature.token_to_orig_map or end_index in feature.tok_end_to_orig_index):
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_ans_len:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "start", "position"]
        )

        seen_predictions = {}
        nbest = []
        null_start = 0
        for pred in prelim_predictions:
            if len(nbest) >= nbest_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                if hasattr(tokenizer, "do_lower_case"):
                    do_lower_case = tokenizer.do_lower_case
                elif hasattr(tokenizer, "basic_tokenizer"):
                    do_lower_case = tokenizer.basic_tokenizer.do_lower_case
                else:
                    do_lower_case = False

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                position = example.positions[orig_doc_start: (orig_doc_end + 1)]

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True
                orig_doc_start = "0"
                position = []

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start=orig_doc_start,
                    position=position
                )
            )
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit, start=null_start))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start=null_start))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start=null_start))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start"] = int(entry.start)
            output['position'] = entry.position
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.uid] = nbest_json[0]["text"]
        else:
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.uid] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.uid] = ""
            else:
                all_predictions[example.uid] = best_non_null_entry.text

        all_nbest_json[example.uid] = nbest_json

    return all_predictions, all_nbest_json


def fastqa_compute_predictions_logits(  # noqa: C901
        all_examples,
        all_features,
        all_results,
        tokenizer,
        max_ans_len,
        nbest_size,
        null_score_diff_threshold,
        version_2_with_negative,
        tokenizer_type="wordpiece",
        verbose_logging=False,
):
    example_q_id_to_features = collections.defaultdict(list)
    for feature in all_features:
        for q_id in feature.q_ids:
            example_q_id_to_features[q_id].append(feature)

    q_id_to_result = {result.uid: result for result in all_results}
    # result.uid = feature.uid + ' ' + q_id

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "q_id", "feature_id", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example in tqdm(all_examples, desc="Compute predictions from logits"):

        for q_id in example.q_ids:
            # print("q_id: ", q_id)
            features = example_q_id_to_features[q_id]

            # Some examples might fail to parse into features. There's no predictions for these examples.
            if len(features) == 0:
                continue

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            null_start = 0
            for (feature_index, feature) in enumerate(features):
                result_id = feature.uid + ' ' + q_id
                result = q_id_to_result[result_id]
                start_indexes = _get_best_indexes(result.start_logits, nbest_size)
                end_indexes = _get_best_indexes(result.end_logits, nbest_size)

                # if we could have irrelevant answers, get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = result.start_logits[0] + result.end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.input_ids):
                            continue
                        if end_index >= len(feature.input_ids):
                            continue
                        if not (
                                start_index in feature.token_to_orig_map or start_index in feature.tok_start_to_orig_index):
                            continue
                        if not (end_index in feature.token_to_orig_map or end_index in feature.tok_end_to_orig_index):
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_ans_len:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                q_id=q_id,
                                feature_id=feature.uid,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                            )
                        )
            # print("compute prelim_predictions cost: ", datetime.datetime.now() - now)
            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        feature_id=0,
                        q_id=q_id,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True,
            )

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit", "start", "position"]
            )

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= nbest_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    # if tokenizer_type != "sentencepiece":
                    tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                    tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                    # tok_text = " ".join(tok_tokens)
                    #
                    # # De-tokenize WordPieces that have been split off.
                    # tok_text = tok_text.replace(" ##", "")
                    # tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    if hasattr(tokenizer, "do_lower_case"):
                        do_lower_case = tokenizer.do_lower_case
                    elif hasattr(tokenizer, "basic_tokenizer"):
                        do_lower_case = tokenizer.basic_tokenizer.do_lower_case
                    else:
                        do_lower_case = False

                    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

                    if final_text in seen_predictions:
                        continue

                    position = example.positions[orig_doc_start: (orig_doc_end + 1)]

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True
                    orig_doc_start = "0"

                    position = []

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        start=orig_doc_start,
                        position=position
                    )
                )
            # if we didn't include the empty option in the n-best, include it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit,
                                                  start=null_start, position=[]))

                # In very rare edge cases we could only have single null prediction.
                # So we just create a nonce prediction in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start=null_start,
                                                     position=[]))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start=null_start, position=[]))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output["start"] = int(entry.start)
                output['position'] = entry.position
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            if not version_2_with_negative:
                all_predictions[q_id] = nbest_json[0]["text"]
            else:
                score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
                scores_diff_json[q_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[q_id] = ""
                else:
                    all_predictions[q_id] = best_non_null_entry.text

            all_nbest_json[q_id] = nbest_json

    return all_predictions, all_nbest_json


def compute_predictions_log_probs(
        all_examples,
        all_features,
        all_results,
        tokenizer,
        max_ans_len,
        nbest_size,
        start_n_top,
        end_n_top,
        verbose_logging=False,
):
    """XLNet compute prediction logic (more complex than Bert's)."""
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"],
    )

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
    )

    example_id_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_id_to_features[feature.example_uid].append(feature)

    id_to_result = {}
    for result in all_results:
        id_to_result[result.uid] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example in tqdm(all_examples, desc="Compute predictions from log probs"):
        features = example_id_to_features[example.uid]

        # Some examples might fail to parse into features. There's no predictions for these examples.
        if len(features) == 0:
            continue

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = id_to_result[feature.uid]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_logits[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_logits[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_ans_len:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob,
                        )
                    )

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True,
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= nbest_size:
                break
            feature = features[pred.feature_index]

            start_orig_pos = feature.tok_start_to_orig_index[pred.start_index]
            end_orig_pos = feature.tok_end_to_orig_index[pred.end_index]

            final_text = example.context_text[start_orig_pos: end_orig_pos + 1].strip()

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob,
                )
            )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.uid] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.uid] = best_non_null_entry.text

        all_nbest_json[example.uid] = nbest_json

    return all_predictions, all_nbest_json


def ner_compute_predictions_logits(  # noqa: C901
        all_features,
        all_results,
        tokenizer
):
    example_id_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_id_to_features[feature.example_uid].append(feature)

    # To compute the overlap of the context, we use the average of all prediction
    id_to_result = {result.uid: result for result in all_results}
    ner_prediction = {}

    for example_id in example_id_to_features:
        features = example_id_to_features[example_id]
        sorted_features = sorted(features, key=lambda x: x.paragraph_start)
        # construct the average logits
        average_logits = None
        full_token_to_original_map = {}
        for i in range(len(sorted_features)):
            feature = sorted_features[i]
            result = id_to_result[feature.uid]
            result_logits = result.logits[
                            1:feature.paragraph_len + 1].numpy()  # [[CLS],......] then ignore the fist token
            feature_token_to_original_map = {j - 1: feature.token_to_orig_map[j] for j in
                                             range(1, feature.paragraph_len + 1)}
            if i == 0:
                average_logits = result_logits
                full_token_to_original_map = feature_token_to_original_map
            else:
                duplicate_part = average_logits.shape[0] - feature.paragraph_start
                for j in range(feature.paragraph_len):  # just for debugging, will remove in production
                    if j < duplicate_part:
                        assert full_token_to_original_map[feature.paragraph_start + j] == feature_token_to_original_map[
                            j]
                    else:
                        full_token_to_original_map[feature.paragraph_start + j] = feature_token_to_original_map[j]
                average_logits[feature.paragraph_start:] = (average_logits[feature.paragraph_start:] * i
                                                            + result_logits[:duplicate_part]) / (i + 1)
                average_logits = np.concatenate((average_logits, result_logits[duplicate_part:]), axis=0)

        assert len(full_token_to_original_map) == average_logits.shape[0]
        # Done calculate ner logits, now decode to get the final entities
        ner_prediction[example_id] = {'logits': average_logits, 'token_to_original_map': full_token_to_original_map}
    return ner_prediction


def squad_evaluate(examples, preds, qa_mode, no_answer_probs=None, no_answer_probability_threshold=1.0):
    assert qa_mode in ['FastQA', 'NormalQA']
    if qa_mode == 'FastQA':
        ground_truth = {}
        for example in examples:
            for i, id in enumerate(example.q_ids):
                ground_truth[id] = example.answers[i]
        qas_id_to_has_answer = {id: bool(ground_truth[id]) for id in ground_truth}
    else:
        qas_id_to_has_answer = {example.uid: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds, qa_mode)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation


def combine_ner_and_QA_result(ner_predictions, qa_results, test_examples):
    q_id_to_example = {qid: example for example in test_examples for qid in example.q_ids}
    q_id_to_question = {example.q_ids[i]: example.question_texts[i] for example in test_examples for i in
                        range(len(example.q_ids))}
    id_to_example = {example.uid for example in test_examples}
    for q_id in qa_results:
        qa_pred = qa_results[q_id][0]
        if qa_pred['text'] in ['', 'empty']:
            continue
        example = q_id_to_example[q_id]
        question = q_id_to_question[q_id]
        qa_entity = {'text': qa_pred['text'], 'entity_type': question, 'start': qa_pred['start']}
        ner_result = ner_predictions[example.uid]
        if qa_entity not in ner_result:
            ner_result.append(qa_entity)
    return ner_predictions


from difflib import SequenceMatcher


def ner_evaluate_by_char(ner_predictions, examples, question_list):
    # calculate ner by field
    id_to_example = {example.uid: example for example in examples}
    confusion_matrix = {question: {'tp': 0, 'fn': 0, 'fp': 0, 'support': 0} for question in question_list}
    for id in id_to_example:
        example = id_to_example[id]
        char_to_word_offset = example.char_to_word_offset
        example_predictions = ner_predictions[id]
        questions = example.question_texts
        for i in range(len(questions)):
            question = questions[i]
            gt = []
            for answer in example.answers[i]:
                gt.append({'answer_start': char_to_word_offset[answer['answer_start']], 'text': answer['text']})
            preds = [{'answer_start': x['start'], 'text': x['text']} for x in example_predictions if
                     x['entity_type'] == question]
            # Remove duplication here

            new_preds = []
            for p in preds:
                if p not in new_preds:
                    new_preds.append(p)
            preds = new_preds
            # Check the overlap here

            for pred in preds:
                # Check if there is any overlapped section
                pred_match_found = False
                for ca in gt:
                    if ca['answer_start'] <= pred['answer_start'] < ca['answer_start'] + \
                            len(ca['text'].strip().split()) or pred['answer_start'] <= ca['answer_start'] < \
                            pred['answer_start'] + len(pred['text'].strip().split()):
                        match = SequenceMatcher(None, ca['text'], pred['text']).find_longest_match(alo=0, ahi=None, blo=0, bhi=None)
                        match_len = match.size
                        tp = match_len / len(pred['text'])
                        assert tp > 0
                        fp = 1 - tp
                        confusion_matrix[question]['tp'] += tp
                        confusion_matrix[question]['fp'] += fp
                        fn = 1 - match_len / len(ca['text'])
                        confusion_matrix[question]['fn'] += fn
                        pred_match_found = True
                        pass
                if not pred_match_found:
                    confusion_matrix[question]['fp'] += 1
            for ca in gt:
                ca_match_found = False
                confusion_matrix[question]['support'] += 1
                for pred in preds:
                    if ca['answer_start'] <= pred['answer_start'] <= ca['answer_start'] + \
                            len(ca['text'].strip().split()) or pred['answer_start'] <= ca['answer_start'] <= \
                            pred['answer_start'] + len(pred['text'].strip().split()):
                        ca_match_found = True
                        break
                if not ca_match_found:
                    confusion_matrix[question]['fn'] += 1
    # calculate accuracy by field
    print("confusion_matrix: ", confusion_matrix)
    for question in confusion_matrix:
        p_denominator = confusion_matrix[question]['tp'] + confusion_matrix[question]['fp']
        if confusion_matrix[question]['support'] == 0:
            if p_denominator == 0:
                confusion_matrix[question]['r'] = 1
                confusion_matrix[question]['p'] = 1
                confusion_matrix[question]['f'] = 1
            else:
                confusion_matrix[question]['r'] = 0
                confusion_matrix[question]['p'] = 0
                confusion_matrix[question]['f'] = 0
        else:
            r_denominator = confusion_matrix[question]['tp'] + confusion_matrix[question]['fn']
            r = confusion_matrix[question]['tp'] / r_denominator
            p = confusion_matrix[question]['tp'] / p_denominator if p_denominator != 0 else 0
            confusion_matrix[question]['r'] = r
            confusion_matrix[question]['p'] = p
            confusion_matrix[question]['f'] = 2 * r * p / (r + p) if (r + p) != 0 else 0
    # average macro
    average = {x: sum([confusion_matrix[question][x] for question in question_list]) / len(question_list) for x in
               ['r', 'p', 'f']}
    macro = {'macro_ner_r': average['r'], 'macro_ner_p': average['p'], 'macro_ner_f1': average['f']}
    # average micro
    micro = {x: sum([confusion_matrix[question][x] for question in question_list]) for x in
             ['tp', 'fp', 'fn', 'support']}
    if micro['support'] == 0:
        micro['micro_ner_r'] = -1
        micro['micro_ner_p'] = -1
        micro['micro_ner_f1'] = -1
    else:
        micro_p_denominator = micro['tp'] + micro['fp']
        micro_r_denominator = micro['tp'] + micro['fn']
        r = micro['tp'] / micro_r_denominator
        p = micro['tp'] / micro_p_denominator if micro_p_denominator != 0 else 0
        micro['micro_ner_r'] = r
        micro['micro_ner_p'] = p
        micro['micro_ner_f1'] = 2 * r * p / (r + p) if (r + p) != 0 else 0
    return (macro, micro, confusion_matrix)


def ner_evaluate(ner_predictions, examples, question_list):
    # calculate ner by field
    id_to_example = {example.uid: example for example in examples}
    confusion_matrix = {question: {'tp': 0, 'fn': 0, 'fp': 0, 'support': 0} for question in question_list}
    for id in id_to_example:
        example = id_to_example[id]
        char_to_word_offset = example.char_to_word_offset
        example_predictions = ner_predictions[id]
        questions = example.question_texts
        for i in range(len(questions)):
            question = questions[i]
            gt = []
            for answer in example.answers[i]:
                gt.append({'answer_start': char_to_word_offset[answer['answer_start']], 'text': answer['text']})
            preds = [{'answer_start': x['start'], 'text': x['text']} for x in example_predictions if
                     x['entity_type'] == question]
            new_preds = []
            for p in preds:
                if p not in new_preds:
                    new_preds.append(p)
            preds = new_preds
            # Remove duplication here
            for pred in preds:
                if pred in gt:
                    confusion_matrix[question]['tp'] += 1
                else:
                    confusion_matrix[question]['fp'] += 1
            for answer in gt:
                confusion_matrix[question]['support'] += 1
                if {'answer_start': answer['answer_start'], 'text': answer['text']} not in preds:
                    confusion_matrix[question]['fn'] += 1
    # calculate accuracy by field
    for question in confusion_matrix:
        p_denominator = confusion_matrix[question]['tp'] + confusion_matrix[question]['fp']
        if confusion_matrix[question]['support'] == 0:
            if p_denominator == 0:
                confusion_matrix[question]['r'] = 1
                confusion_matrix[question]['p'] = 1
                confusion_matrix[question]['f'] = 1
            else:
                confusion_matrix[question]['r'] = 0
                confusion_matrix[question]['p'] = 0
                confusion_matrix[question]['f'] = 0
        else:
            r_denominator = confusion_matrix[question]['tp'] + confusion_matrix[question]['fn']
            r = confusion_matrix[question]['tp'] / r_denominator
            p = confusion_matrix[question]['tp'] / p_denominator if p_denominator != 0 else 0
            confusion_matrix[question]['r'] = r
            confusion_matrix[question]['p'] = p
            confusion_matrix[question]['f'] = 2 * r * p / (r + p) if (r + p) != 0 else 0
    # average macro
    average = {x: sum([confusion_matrix[question][x] for question in question_list]) / len(question_list) for x in
               ['r', 'p', 'f']}
    macro = {'macro_ner_r': average['r'], 'macro_ner_p': average['p'], 'macro_ner_f1': average['f']}
    # average micro
    micro = {x: sum([confusion_matrix[question][x] for question in question_list]) for x in
             ['tp', 'fp', 'fn', 'support']}
    if micro['support'] == 0:
        micro['micro_ner_r'] = -1
        micro['micro_ner_p'] = -1
        micro['micro_ner_f1'] = -1
    else:
        micro_p_denominator = micro['tp'] + micro['fp']
        micro_r_denominator = micro['tp'] + micro['fn']
        r = micro['tp'] / micro_r_denominator
        p = micro['tp'] / micro_p_denominator if micro_p_denominator != 0 else 0
        micro['micro_ner_r'] = r
        micro['micro_ner_p'] = p
        micro['micro_ner_f1'] = 2 * r * p / (r + p) if (r + p) != 0 else 0
    return (macro, micro, confusion_matrix)
