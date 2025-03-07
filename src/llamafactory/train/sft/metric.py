# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Union, List, Any

import re
import numpy as np
import torch
import torch.nn.functional as F

from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        # decode text
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # accuray f1 

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


@dataclass
class ComputeExactMatch:
    r"""
    Computes ExactMatch and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"exact_match": []}
        print(f"{result=}")
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        for pred, label in zip(decoded_preds, decoded_labels):
            print(f"pred : {pred}")
            print(f"label: {label}")
            print("\n")
            if pred == label:
                self.score_dict["exact_match"].append(1)
            else:
                self.score_dict["exact_match"].append(0)
            print(f"pred == label, {pred == label}")
            
        if compute_result:
            print(f"Sample Count: {len(preds)=}")
            return self._dump()


@dataclass
class ComputeRegressionMetrics:
    """
    Computes regression metrics and supports `batch_eval_metrics`.
    Extracts numerical values from model outputs and labels for regression tasks.
    """
    
    tokenizer: "PreTrainedTokenizer"
    
    def __post_init__(self):
        self._dump()
    
    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            print(f"{result=}")
        self.score_dict = {
            "mae": [],
            "rmse": [],
            "spearman_corr": [],
            "pearson_corr": []
        }
        return result
    
    
    def extract_value_from_text(self, text: str) -> float:
        """
        Extract numerical value from text, specifically looking for content between <answer> and </answer> tags.
        If no valid number is found, return 0.0.
        """
        # Find content between <answer> and </answer> tags
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        answer_match = answer_pattern.search(text)
        
        if answer_match:
            content = answer_match.group(1).strip()
            # Try to extract a number (float or int) from the content
            number_pattern = re.compile(r'[-+]?\d*\.?\d+')
            number_match = number_pattern.search(content)
            if number_match:
                num_str = number_match.group(0)
                # Convert to float (works for both integers and floats)
                return float(num_str)
        
        return 0.0
    
    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        
        # print(f"{eval_preds=}")
        # exit()
        
        # Extract predictions and labels
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        
        # Extract numerical values from decoded text
        pred_values = []
        label_values = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            print(f"pred : {pred}")
            print(f"label: {label}")
            print("\n")
            # Extract values
            pred_value = self.extract_value_from_text(pred)
            label_value = self.extract_value_from_text(label)
            
            pred_values.append(pred_value)
            label_values.append(label_value)
            
            # Individual metrics
            self.score_dict["mae"].append(abs(pred_value - label_value))
            self.score_dict["rmse"].append((pred_value - label_value)**2)
            
            print(f"Extracted: pred={pred_value}, label={label_value}, diff={abs(pred_value - label_value)}")
            print("\n")
            
        # Convert to numpy arrays for batch calculations
        pred_values = np.array(pred_values)
        label_values = np.array(label_values)
        
        # Compute dataset-level metrics
        # RMSE needs to be square-rooted after averaging
        self.score_dict["rmse"] = [np.sqrt(np.mean(self.score_dict["rmse"]))]
        
        # Compute correlations for the entire batch
        if len(pred_values) > 1:  # Correlation requires at least 2 points
            try:
                spearman = spearmanr(pred_values, label_values)[0]
                if np.isnan(spearman):
                    spearman = 0.0
                self.score_dict["spearman_corr"] = [spearman]
            except:
                self.score_dict["spearman_corr"] = [0.0]
                
            try:
                pearson = pearsonr(pred_values, label_values)[0]
                if np.isnan(pearson):
                    pearson = 0.0
                self.score_dict["pearson_corr"] = [pearson]
            except:
                self.score_dict["pearson_corr"] = [0.0]
        else:
            # Cannot compute correlation with a single sample
            self.score_dict["spearman_corr"] = [0.0]
            self.score_dict["pearson_corr"] = [0.0]
            
        if compute_result:
            print(f"Sample Count: {len(preds)=}")
            return self._dump()


@dataclass
class ComputeAucMetrics:
    """
    Computes AUC for Yes/No answers within <answer> tags and supports batch evaluation.
    
    This metrics computer works by finding Yes/No answers in <answer> tags in the generated text,
    extracting the logits for Yes/No tokens, and computing AUC based on the true labels.
    """
    tokenizer: Any
    yes_token_id: Optional[int] = None
    no_token_id: Optional[int] = None
    score_dict: Dict[str, List[float]] = field(default_factory=lambda: {"auc": []})
    
    def _dump(self) -> Optional[Dict[str, float]]:
        """Compute final metrics from accumulated scores and reset score_dict."""
        result = None
        if hasattr(self, "score_dict") and self.score_dict["auc"]:
            # Calculate mean of the scores if we have any
            result = {"auc": float(np.mean(self.score_dict["auc"]))}
        
        # Reset scores for next evaluation
        self.score_dict = {"auc": []}
        return result
    
    def __post_init__(self):
        """Initialize token IDs and reset scores after initialization."""
        # Get token IDs for "Yes" and "No" if not provided
        if self.yes_token_id is None:
            self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        if self.no_token_id is None:
            self.no_token_id = self.tokenizer.convert_tokens_to_ids("No")
        
        # If using a subword tokenizer, we might need to get the full word tokens
        if self.yes_token_id == self.tokenizer.unk_token_id:
            # Try different cases or with leading space
            potential_yes_tokens = ["Yes", "yes", " Yes", " yes"]
            potential_no_tokens = ["No", "no", " No", " no"]
            
            for token in potential_yes_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id:
                    self.yes_token_id = token_id
                    break
            
            for token in potential_no_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.tokenizer.unk_token_id:
                    self.no_token_id = token_id
                    break
        
        # Reset the score dictionary
        self._dump()
    
    def extract_answer_tag_content(self, text: str) -> str:
        """Extract content within the <answer> tag from text."""
        import re
        match = re.search(r'<answer>\s*(\w+)\s*</answer>', text)
        if match:
            return match.group(1).strip()
        return ""
    
    def extract_label_from_ids(self, label_ids) -> int:
        """Extract label (0=No, 1=Yes) from label token IDs by finding <answer> tag."""
        # Decode the label token IDs to text
        decoded_label = self.tokenizer.decode(label_ids, skip_special_tokens=True)
        
        # Extract content within <answer> tag
        answer_content = self.extract_answer_tag_content(decoded_label)
        
        # Determine label value
        if answer_content.lower() == "yes":
            return 1
        elif answer_content.lower() == "no":
            return 0
        else:
            # If no clear answer found, use a default value
            return -1  # or some appropriate default
    
    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        """
        Process each prediction and compute AUC.
        
        Args:
            eval_preds: Predictions and labels
            compute_result: Whether to calculate and return final results
            
        Returns:
            Dictionary with metrics if compute_result is True, otherwise None
        """
        predictions, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        
        # Handle different prediction formats
        if isinstance(predictions, tuple) and len(predictions) >= 2:
            logits, _ = predictions
        else:
            logits = predictions[0] if isinstance(predictions, tuple) else predictions
        
        # Process each sample
        for i in range(len(logits)):
            # Skip if we're out of bounds
            if i >= len(labels):
                continue
                
            # Extract label from the label IDs
            label_ids = labels[i]
            # Filter out padding tokens if necessary
            valid_label_ids = [id for id in label_ids if id != self.tokenizer.pad_token_id]
            true_label = self.extract_label_from_ids(valid_label_ids)
            
            # Skip samples where label couldn't be determined
            if true_label == -1:
                continue
            
            # Decode prediction to text to find <answer> tag
            pred_tokens = np.argmax(logits[i], axis=-1)
            decoded_pred = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            answer_content = self.extract_answer_tag_content(decoded_pred)
            
            # Default probability if we can't determine
            yes_prob = 0.5
            
            # If Yes or No is directly extracted from text
            if answer_content.lower() == "yes":
                yes_prob = 1.0
            elif answer_content.lower() == "no":
                yes_prob = 0.0
            else:
                # Try to get probabilities from logits
                # Encode the text to find position of <answer> tag
                encoded_pred = self.tokenizer.encode(decoded_pred, add_special_tokens=False)
                answer_tag_tokens = self.tokenizer.encode("<answer>", add_special_tokens=False)
                
                try:
                    for j in range(len(encoded_pred) - len(answer_tag_tokens)):
                        if encoded_pred[j:j+len(answer_tag_tokens)] == answer_tag_tokens:
                            # Found position after <answer> tag
                            position = j + len(answer_tag_tokens)
                            
                            # Get logits at this position
                            if position < logits[i].shape[0]:
                                # Get logits for Yes and No tokens
                                yes_logit = logits[i, position, self.yes_token_id]
                                no_logit = logits[i, position, self.no_token_id]
                                
                                # Apply softmax considering only these two tokens
                                scores = np.array([no_logit, yes_logit])
                                yes_prob = F.softmax(torch.tensor(scores), dim=0)[1].item()
                                break
                except Exception as e:
                    print(f"Error processing prediction {i}: {e}")
            
            # Calculate AUC for this single sample (either 0 or 1)
            # For single samples, AUC is 1 if prediction is correct, 0 if incorrect, 0.5 if undecided
            single_sample_auc = 1.0 if (yes_prob > 0.5 and true_label == 1) or (yes_prob < 0.5 and true_label == 0) else 0.0
            
            # Accumulate the score
            self.score_dict["auc"].append(single_sample_auc)
        
        # If we have enough samples, calculate AUC across all samples
        if len(self.score_dict["auc"]) >= 2:
            # Collect all yes_probs and true_labels for proper AUC calculation
            yes_probs = []
            true_labels = []
            
            # Process each sample again for proper AUC
            for i in range(len(logits)):
                if i >= len(labels):
                    continue
                    
                label_ids = labels[i]
                valid_label_ids = [id for id in label_ids if id != self.tokenizer.pad_token_id]
                true_label = self.extract_label_from_ids(valid_label_ids)
                
                if true_label == -1:
                    continue
                
                pred_tokens = np.argmax(logits[i], axis=-1)
                decoded_pred = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                answer_content = self.extract_answer_tag_content(decoded_pred)
                
                yes_prob = 0.5
                
                if answer_content.lower() == "yes":
                    yes_prob = 1.0
                elif answer_content.lower() == "no":
                    yes_prob = 0.0
                else:
                    # Same logic to extract probability from logits as above
                    encoded_pred = self.tokenizer.encode(decoded_pred, add_special_tokens=False)
                    answer_tag_tokens = self.tokenizer.encode("<answer>", add_special_tokens=False)
                    
                    try:
                        for j in range(len(encoded_pred) - len(answer_tag_tokens)):
                            if encoded_pred[j:j+len(answer_tag_tokens)] == answer_tag_tokens:
                                position = j + len(answer_tag_tokens)
                                
                                if position < logits[i].shape[0]:
                                    yes_logit = logits[i, position, self.yes_token_id]
                                    no_logit = logits[i, position, self.no_token_id]
                                    
                                    scores = np.array([no_logit, yes_logit])
                                    yes_prob = F.softmax(torch.tensor(scores), dim=0)[1].item()
                                    break
                    except Exception:
                        pass
                
                yes_probs.append(yes_prob)
                true_labels.append(true_label)
            
            # Calculate proper AUC if we have enough samples
            if len(yes_probs) >= 2 and len(set(true_labels)) > 1:
                try:
                    batch_auc = roc_auc_score(true_labels, yes_probs)
                    # Replace individual sample scores with the proper batch AUC
                    self.score_dict["auc"] = [batch_auc]
                except ValueError as e:
                    print(f"AUC calculation error: {e}")
        
        if compute_result:
            return self._dump()
        return None
