from unsloth import FastModel
import os
import json
import torch
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset



class GemmaFineTuning:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        self.training_history = {"loss": [], "eval_loss": [], "step": []}
        self.model_save_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.fourbit_models = [
            "unsloth/gemma-2b-it-4bit",
            "unsloth/gemma-7b-it-4bit",
            "unsloth/gemma-2b-4bit",
            "unsloth/gemma-7b-4bit"
        ]
        # Default hyperparameters
        self.default_params = {
            "model_name": "unsloth/gemma-2b-it-4bit",
            "learning_rate": 2e-5,
            "batch_size": 8,
            "epochs": 3,
            "max_length": 512,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "eval_ratio": 0.1,
        }

    def load_model_and_tokenizer(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer"""
        try:
            # Map UI model names to actual model IDs
            model_mapping = {
                "google/gemma-2b": "unsloth/gemma-2b-4bit",
                "google/gemma-7b": "unsloth/gemma-7b-4bit",
                "google/gemma-2b-it": "unsloth/gemma-2b-it-4bit",
                "google/gemma-7b-it": "unsloth/gemma-7b-it-4bit"
            }

            actual_model_name = model_mapping.get(model_name, model_name)
            
            model, tokenizer = FastModel.from_pretrained(
                model_name=actual_model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                load_in_8bit=False,
                full_finetuning=False,
            )

            # Move model to device
            model = model.to(self.device)
            return model, tokenizer

        except Exception as e:
            raise ValueError(f"Error loading model {model_name}: {str(e)}")

    def prepare_dataset(self, file_path, format_type):
        """
        Prepare and normalize dataset from various formats

        Parameters:
        file_path (str): Path to the dataset file
        format_type (str): File format (csv, jsonl, text)

        Returns:
        dict: Dataset dictionary with train split
        """
        import pandas as pd
        import json
        import os
        from datasets import Dataset, DatasetDict

        try:
            if format_type == "csv":
                # Load CSV file
                df = pd.read_csv(file_path)

                # Check if the CSV has the expected columns (looking for either instruction-response pairs or text)
                if "instruction" in df.columns and "response" in df.columns:
                    # Instruction-following dataset format
                    dataset_format = "instruction-response"
                    # Ensure no nulls
                    df = df.dropna(subset=["instruction", "response"])
                    # Create formatted text by combining instruction and response
                    df["text"] = df.apply(lambda row: f"<instruction>{row['instruction']}</instruction>\n<response>{row['response']}</response>", axis=1)
                elif "input" in df.columns and "output" in df.columns:
                    # Another common format
                    dataset_format = "input-output"
                    df = df.dropna(subset=["input", "output"])
                    df["text"] = df.apply(lambda row: f"<input>{row['input']}</input>\n<output>{row['output']}</output>", axis=1)
                elif "prompt" in df.columns and "completion" in df.columns:
                    # OpenAI-style format
                    dataset_format = "prompt-completion"
                    df = df.dropna(subset=["prompt", "completion"])
                    df["text"] = df.apply(lambda row: f"<prompt>{row['prompt']}</prompt>\n<completion>{row['completion']}</completion>", axis=1)
                elif "text" in df.columns:
                    # Simple text format
                    dataset_format = "text-only"
                    df = df.dropna(subset=["text"])
                else:
                    # Try to infer format from the first text column
                    text_columns = [col for col in df.columns if df[col].dtype == 'object']
                    if len(text_columns) > 0:
                        dataset_format = "inferred"
                        df["text"] = df[text_columns[0]]
                        df = df.dropna(subset=["text"])
                    else:
                        raise ValueError("CSV file must contain either 'instruction'/'response', 'input'/'output', 'prompt'/'completion', or 'text' columns")

                # Create dataset
                dataset = Dataset.from_pandas(df)

            elif format_type == "jsonl":
                # Load JSONL file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f if line.strip()]

                # Check and normalize the format
                normalized_data = []
                for item in data:
                    normalized_item = {}

                    # Try to find either instruction-response pairs or text
                    if "instruction" in item and "response" in item:
                        normalized_item["text"] = f"<instruction>{item['instruction']}</instruction>\n<response>{item['response']}</response>"
                        normalized_item["instruction"] = item["instruction"]
                        normalized_item["response"] = item["response"]
                    elif "input" in item and "output" in item:
                        normalized_item["text"] = f"<input>{item['input']}</input>\n<output>{item['output']}</output>"
                        normalized_item["input"] = item["input"]
                        normalized_item["output"] = item["output"]
                    elif "prompt" in item and "completion" in item:
                        normalized_item["text"] = f"<prompt>{item['prompt']}</prompt>\n<completion>{item['completion']}</completion>"
                        normalized_item["prompt"] = item["prompt"]
                        normalized_item["completion"] = item["completion"]
                    elif "text" in item:
                        normalized_item["text"] = item["text"]
                    else:
                        # Try to infer from the first string value
                        text_keys = [k for k, v in item.items() if isinstance(v, str) and len(v.strip()) > 0]
                        if text_keys:
                            normalized_item["text"] = item[text_keys[0]]
                        else:
                            continue  # Skip this item if no usable text found

                    normalized_data.append(normalized_item)

                if not normalized_data:
                    raise ValueError("No valid data items found in the JSONL file")

                # Create dataset
                dataset = Dataset.from_list(normalized_data)

            elif format_type == "text":
                # For text files, split by newlines and create entries
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check if it's a single large document or multiple examples
                # If file size > 10KB, try to split into paragraphs
                if os.path.getsize(file_path) > 10240:
                    # Split by double newlines (paragraphs)
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    # Filter out very short paragraphs (less than 20 chars)
                    paragraphs = [p for p in paragraphs if len(p) >= 20]
                    data = [{"text": p} for p in paragraphs]
                else:
                    # Treat as a single example
                    data = [{"text": content}]

                # Create dataset
                dataset = Dataset.from_list(data)
            else:
                raise ValueError(f"Unsupported file format: {format_type}")

            # Return as a DatasetDict with a train split
            return DatasetDict({"train": dataset})

        except Exception as e:
            import traceback
            error_msg = f"Error processing dataset: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise ValueError(error_msg)

    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size characters"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for the space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def preprocess_dataset(self, dataset, tokenizer, max_length):
        """
        Tokenize and format the dataset for training

        Parameters:
        dataset (DatasetDict): Dataset dictionary with train and validation splits
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length

        Returns:
        DatasetDict: Tokenized dataset ready for training
        """
        def tokenize_function(examples):
            # Check if the dataset has both input and target text columns
            if "text" in examples:
                texts = examples["text"]
                inputs = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs["labels"] = inputs["input_ids"].clone()
                return inputs
            else:
                # Try to find text columns based on common naming patterns
                potential_text_cols = [col for col in examples.keys() if isinstance(examples[col], list) and
                                    all(isinstance(item, str) for item in examples[col])]

                if not potential_text_cols:
                    raise ValueError("No suitable text columns found in the dataset")

                # Use the first text column found
                text_col = potential_text_cols[0]
                texts = examples[text_col]

                inputs = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs["labels"] = inputs["input_ids"].clone()
                return inputs

        # Apply tokenization to each split
        tokenized_dataset = {}
        for split, ds in dataset.items():
            tokenized_dataset[split] = ds.map(
                tokenize_function,
                batched=True,
                remove_columns=ds.column_names
            )

        return tokenized_dataset

    def prepare_training_args(self, params: Dict) -> TrainingArguments:
        """Set up training arguments based on hyperparameters"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_save_path = f"gemma-finetuned-{timestamp}"

        args = TrainingArguments(
            output_dir=self.model_save_path,
            per_device_train_batch_size=params.get("batch_size", self.default_params["batch_size"]),
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=params.get("batch_size", self.default_params["batch_size"]),
            learning_rate=params.get("learning_rate", self.default_params["learning_rate"]),
            num_train_epochs=params.get("epochs", self.default_params["epochs"]),
            warmup_ratio=params.get("warmup_ratio", self.default_params["warmup_ratio"]),
            weight_decay=params.get("weight_decay", self.default_params["weight_decay"]),
            logging_steps=1,
            evaluation_strategy="steps" if params.get("eval_ratio", 0) > 0 else "no",
            eval_steps=100 if params.get("eval_ratio", 0) > 0 else None,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True if params.get("eval_ratio", 0) > 0 else False,
            report_to="none"
        )
        return args

    def train(self, training_params: Dict) -> str:
        """Main training method that handles the complete training pipeline"""
        try:
            if self.dataset is None:
                return "Error: No dataset loaded. Please preprocess a dataset first."

            # Reset training history
            self.training_history = {"loss": [], "eval_loss": [], "step": []}

            # Load model and tokenizer if not already loaded or if model name changed
            current_model_name = training_params.get("model_name", self.default_params["model_name"])
            if (self.model is None or self.tokenizer is None or 
                getattr(self, '_current_model_name', None) != current_model_name):
                
                self.model, self.tokenizer = self.load_model_and_tokenizer(current_model_name)
                self._current_model_name = current_model_name

            # Create validation split if needed
            eval_ratio = float(training_params.get("eval_ratio", self.default_params["eval_ratio"]))
            if eval_ratio > 0 and "validation" not in self.dataset:
                split_dataset = self.dataset["train"].train_test_split(test_size=eval_ratio)
                self.dataset = {
                    "train": split_dataset["train"],
                    "validation": split_dataset["test"]
                }

            # Apply LoRA if selected
            if training_params.get("use_lora", self.default_params["use_lora"]):
                self.model = self.setup_lora(self.model, {
                    "lora_r": int(training_params.get("lora_r", self.default_params["lora_r"])),
                    "lora_alpha": int(training_params.get("lora_alpha", self.default_params["lora_alpha"])),
                    "lora_dropout": float(training_params.get("lora_dropout", self.default_params["lora_dropout"]))
                })

            # Preprocess dataset
            max_length = int(training_params.get("max_length", self.default_params["max_length"]))
            tokenized_dataset = self.preprocess_dataset(self.dataset, self.tokenizer, max_length)

            # Update training arguments with proper type conversion
            training_args = self.prepare_training_args({
                "batch_size": int(training_params.get("batch_size", self.default_params["batch_size"])),
                "learning_rate": float(training_params.get("learning_rate", self.default_params["learning_rate"])),
                "epochs": int(training_params.get("epochs", self.default_params["epochs"])),
                "weight_decay": float(training_params.get("weight_decay", self.default_params["weight_decay"])),
                "warmup_ratio": float(training_params.get("warmup_ratio", self.default_params["warmup_ratio"])),
                "eval_ratio": eval_ratio
            })

            # Create trainer with proper callback
            self.trainer = self.create_trainer(
                self.model,
                self.tokenizer,
                tokenized_dataset,
                training_args
            )

            # Start training
            self.trainer.train()
            
            # Save the model
            save_path = f"models/gemma-finetuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(save_path, exist_ok=True)
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.model_save_path = save_path

            return f"Training completed successfully! Model saved to {save_path}"

        except Exception as e:
            import traceback
            return f"Error during training: {str(e)}\n{traceback.format_exc()}"

    def setup_lora(self, model, params: Dict) -> torch.nn.Module:
        """Configure LoRA for parameter-efficient fine-tuning"""
        # Prepare the model for training if using 8-bit or 4-bit quantization
        if hasattr(model, "is_quantized") and model.is_quantized:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=params["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = FastModel.get_peft_model(
                model,
                finetune_vision_layers     = False, # Turn off for just text!
                finetune_language_layers   = True,  # Should leave on!
                finetune_attention_modules = True,  # Attention good for GRPO
                finetune_mlp_modules       = True,  # SHould leave on always!

                r = 8,           # Larger = higher accuracy, but might overfit
                lora_alpha = 8,  # Recommended alpha == r at least
                lora_dropout = 0,
                bias = "none",
                random_state = 3407,
            )
        model.print_trainable_parameters()
        model = model.to(self.device)
        return model

    def create_trainer(self, model, tokenizer, dataset, training_args):
        """Set up the Trainer for model fine-tuning"""
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Custom callback to store training history and progress
        class TrainingCallback(TrainerCallback):
            def __init__(self, app):
                self.app = app
                self.total_steps = 0
                self.current_step = 0
                self.current_loss = 0
                self.total_loss = 0
                self.step_count = 0

            def on_train_begin(self, args, state, control, **kwargs):
                """Called at the beginning of training"""
                train_dl_size = len(state.train_dataloader)
                self.total_steps = int(train_dl_size * args.num_train_epochs)
                self.app.training_progress = {
                    "total_steps": self.total_steps,
                    "current_step": 0,
                    "current_loss": 0.0,
                    "avg_loss": 0.0,
                    "progress_percentage": 0.0
                }

            def on_step_end(self, args, state, control, **kwargs):
                """Called at the end of each step"""
                self.current_step = state.global_step
                progress = (self.current_step / self.total_steps) * 100
                self.app.training_progress["current_step"] = self.current_step
                self.app.training_progress["progress_percentage"] = progress

            def on_log(self, args, state, control, logs=None, **kwargs):
                """Called when logs are written"""
                if logs:
                    if "loss" in logs:
                        self.current_loss = logs["loss"]
                        self.total_loss += logs["loss"]
                        self.step_count += 1
                        avg_loss = self.total_loss / self.step_count
                        
                        self.app.training_progress["current_loss"] = self.current_loss
                        self.app.training_progress["avg_loss"] = avg_loss
                        
                        for key in ['loss', 'eval_loss']:
                            if key in logs:
                                self.app.training_history[key].append(logs[key])
                                if 'step' in logs:
                                    self.app.training_history['step'].append(logs['step'])

        # Create trainer with callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else None,
            data_collator=data_collator,
            callbacks=[TrainingCallback(self)]
        )

        return trainer

    def plot_training_progress(self):
        """Generate a plot of the training progress"""
        if not self.training_history["loss"]:
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history["step"], self.training_history["loss"], label="Training Loss")

        if self.training_history["eval_loss"]:
            # Get the steps where eval happened
            eval_steps = self.training_history["step"][:len(self.training_history["eval_loss"])]
            plt.plot(eval_steps, self.training_history["eval_loss"], label="Validation Loss", linestyle="--")

        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        return plt

    def export_model(self, output_format: str) -> str:
        """Export the fine-tuned model in various formats"""
        if self.model is None or self.model_save_path is None:
            return "No model has been trained yet."

        export_path = f"{self.model_save_path}/exported_{output_format}"
        os.makedirs(export_path, exist_ok=True)

        if output_format == "pytorch":
            # Save as PyTorch format
            self.model.save_pretrained(export_path)
            self.tokenizer.save_pretrained(export_path)
            return f"Model exported in PyTorch format to {export_path}"

        elif output_format == "tensorflow":
            # Convert to TensorFlow format
            try:
                from transformers.modeling_tf_utils import convert_pt_to_tf

                # First save the PyTorch model
                self.model.save_pretrained(export_path)
                self.tokenizer.save_pretrained(export_path)

                # Then convert to TF SavedModel format
                tf_model = convert_pt_to_tf(self.model)
                tf_model.save_pretrained(f"{export_path}/tf_saved_model")

                return f"Model exported in TensorFlow format to {export_path}/tf_saved_model"
            except Exception as e:
                return f"Failed to export as TensorFlow model: {str(e)}"

        elif output_format == "gguf":
            # Export as GGUF format for local inference
            try:
                import subprocess

                # First save the model in PyTorch format
                self.model.save_pretrained(export_path)
                self.tokenizer.save_pretrained(export_path)

                # Use llama.cpp's conversion script (must be installed)
                subprocess.run([
                    "python", "-m", "llama_cpp.convert",
                    "--outtype", "gguf",
                    "--outfile", f"{export_path}/model.gguf",
                    export_path
                ])

                return f"Model exported in GGUF format to {export_path}/model.gguf"
            except Exception as e:
                return f"Failed to export as GGUF model: {str(e)}"

        else:
            return f"Unsupported export format: {output_format}"

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the fine-tuned model"""
        if self.model is None or self.tokenizer is None:
            return "No model has been loaded or fine-tuned yet."

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length + inputs.input_ids.shape[1],
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text