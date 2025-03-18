import gradio as gr
from utils.check_dataset import validate_dataset, generate_dataset_report
from utils.sample_dataset import generate_sample_datasets
from utils.model import GemmaFineTuning

class GemmaUI:
    def __init__(self):
        self.model_instance = GemmaFineTuning()
        self.default_params = self.model_instance.default_params

    def create_ui(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Gemma Fine-tuning UI") as app:
            gr.Markdown("# Gemma Model Fine-tuning Interface")
            gr.Markdown("Upload your dataset, configure parameters, and fine-tune a Gemma model")

            with gr.Tabs():
                with gr.TabItem("1. Data Upload & Preprocessing"):
                    with gr.Row():
                        with gr.Column():
                            file_upload = gr.File(label="Upload Dataset")
                            file_format = gr.Radio(
                                ["csv", "jsonl", "text"],
                                label="File Format",
                                value="csv"
                            )
                            preprocess_button = gr.Button("Preprocess Dataset")
                            dataset_info = gr.TextArea(label="Dataset Information", interactive=False)

                with gr.TabItem("2. Model & Hyperparameters"):
                    with gr.Row():
                        with gr.Column():
                            model_name = gr.Dropdown(
                                choices=[
                                    "google/gemma-2b",
                                    "google/gemma-7b",
                                    "google/gemma-2b-it",
                                    "google/gemma-7b-it"
                                ],
                                value=self.default_params["model_name"],
                                label="Model Name",
                                info="Select a Gemma model to fine-tune"
                            )
                            learning_rate = gr.Slider(
                                minimum=1e-6,
                                maximum=5e-4,
                                value=self.default_params["learning_rate"],
                                label="Learning Rate",
                                info="Learning rate for the optimizer"
                            )
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=32,
                                step=1,
                                value=self.default_params["batch_size"],
                                label="Batch Size",
                                info="Number of samples in each training batch"
                            )
                            epochs = gr.Slider(
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=self.default_params["epochs"],
                                label="Epochs",
                                info="Number of training epochs"
                            )

                        with gr.Column():
                            max_length = gr.Slider(
                                minimum=128,
                                maximum=2048,
                                step=16,
                                value=self.default_params["max_length"],
                                label="Max Sequence Length",
                                info="Maximum token length for inputs"
                            )
                            use_lora = gr.Checkbox(
                                value=self.default_params["use_lora"],
                                label="Use LoRA for Parameter-Efficient Fine-tuning",
                                info="Recommended for faster training and lower memory usage"
                            )
                            lora_r = gr.Slider(
                                minimum=4,
                                maximum=64,
                                step=4,
                                value=self.default_params["lora_r"],
                                label="LoRA Rank (r)",
                                info="Rank of the LoRA update matrices",
                                visible=lambda: use_lora.value
                            )
                            lora_alpha = gr.Slider(
                                minimum=4,
                                maximum=64,
                                step=4,
                                value=self.default_params["lora_alpha"],
                                label="LoRA Alpha",
                                info="Scaling factor for LoRA updates",
                                visible=lambda: use_lora.value
                            )
                            eval_ratio = gr.Slider(
                                minimum=0.05,
                                maximum=0.3,
                                step=0.05,
                                value=self.default_params["eval_ratio"],
                                label="Validation Split Ratio",
                                info="Portion of data to use for validation"
                            )

                with gr.TabItem("3. Training"):
                    with gr.Row():
                        with gr.Column():
                            start_training_button = gr.Button("Start Fine-tuning")
                            stop_training_button = gr.Button("Stop Training", variant="stop")
                            training_status = gr.Textbox(label="Training Status", interactive=False)

                        with gr.Column():
                            progress_plot = gr.Plot(label="Training Progress")
                            refresh_plot_button = gr.Button("Refresh Plot")

                with gr.TabItem("4. Evaluation & Export"):
                    with gr.Row():
                        with gr.Column():
                            test_prompt = gr.Textbox(
                                label="Test Prompt",
                                placeholder="Enter a prompt to test the model...",
                                lines=3
                            )
                            max_gen_length = gr.Slider(
                                minimum=10,
                                maximum=500,
                                step=10,
                                value=100,
                                label="Max Generation Length"
                            )
                            generate_button = gr.Button("Generate Text")
                            generated_output = gr.Textbox(label="Generated Output", lines=10, interactive=False)

                        with gr.Column():
                            export_format = gr.Radio(
                                ["pytorch", "tensorflow", "gguf"],
                                label="Export Format",
                                value="pytorch"
                            )
                            export_button = gr.Button("Export Model")
                            export_status = gr.Textbox(label="Export Status", interactive=False)

            # Functionality
            def preprocess_data(file, format_type):
                try:
                    if file is None:
                        return "Please upload a file first."

                    # Process the uploaded file
                    dataset = self.model_instance.prepare_dataset(file.name, format_type)
                    self.model_instance.dataset = dataset

                    # Create a summary of the dataset
                    num_samples = len(dataset["train"])


                    # Sample a few examples
                    examples = dataset["train"].select(range(min(3, num_samples)))
                    sample_text = []
                    for ex in examples:
                        text_key = list(ex.keys())[0] if "text" not in ex else "text"
                        sample = ex[text_key]
                        if isinstance(sample, str):
                            sample_text.append(sample[:100] + "..." if len(sample) > 100 else sample)

                    info = f"Dataset loaded successfully!\n"
                    info += f"Number of training examples: {num_samples}\n"
                    info += f"Sample data:\n" + "\n---\n".join(sample_text)

                    return info
                except Exception as e:
                    return f"Error preprocessing data: {str(e)}"

            def start_training(
                model_name, learning_rate, batch_size, epochs, max_length,
                use_lora, lora_r, lora_alpha, eval_ratio
            ):
                try:
                    if self.model_instance.dataset is None:
                        return "Please preprocess a dataset first."

                    # Validate parameters
                    if not model_name:
                        return "Please select a model."
                    
                    # Prepare training parameters with proper type conversion
                    training_params = {
                        "model_name": str(model_name),
                        "learning_rate": float(learning_rate),
                        "batch_size": int(batch_size),
                        "epochs": int(epochs),
                        "max_length": int(max_length),
                        "use_lora": bool(use_lora),
                        "lora_r": int(lora_r) if use_lora else None,
                        "lora_alpha": int(lora_alpha) if use_lora else None,
                        "eval_ratio": float(eval_ratio),
                        "weight_decay": float(self.default_params["weight_decay"]),
                        "warmup_ratio": float(self.default_params["warmup_ratio"]),
                        "lora_dropout": float(self.default_params["lora_dropout"])
                    }

                    # Start training in a separate thread
                    import threading
                    def train_thread():
                        status = self.model_instance.train(training_params)
                        return status

                    thread = threading.Thread(target=train_thread)
                    thread.start()

                    return "Training started! Monitor the progress in the Training tab."
                except Exception as e:
                    return f"Error starting training: {str(e)}"

            def stop_training():
                if self.model_instance.trainer is not None:
                    # Attempt to stop the trainer
                    self.model_instance.trainer.stop_training = True
                    return "Training stop signal sent. It may take a moment to complete the current step."
                return "No active training to stop."

            def update_progress_plot():
                try:
                    return self.model_instance.plot_training_progress()
                except Exception as e:
                    return None

            def run_text_generation(prompt, max_length):
                try:
                    if self.model_instance.model is None:
                        return "Please fine-tune a model first."

                    return self.model_instance.generate_text(prompt, int(max_length))
                except Exception as e:
                    return f"Error generating text: {str(e)}"

            def export_model_fn(format_type):
                try:
                    if self.model_instance.model is None:
                        return "Please fine-tune a model first."

                    return self.model_instance.export_model(format_type)
                except Exception as e:
                    return f"Error exporting model: {str(e)}"

            # Connect UI components to functions
            preprocess_button.click(
                preprocess_data,
                inputs=[file_upload, file_format],
                outputs=dataset_info
            )

            start_training_button.click(
                start_training,
                inputs=[
                    model_name, learning_rate, batch_size, epochs, max_length,
                    use_lora, lora_r, lora_alpha, eval_ratio
                ],
                outputs=training_status
            )

            stop_training_button.click(
                stop_training,
                inputs=[],
                outputs=training_status
            )

            refresh_plot_button.click(
                update_progress_plot,
                inputs=[],
                outputs=progress_plot
            )

            generate_button.click(
                run_text_generation,
                inputs=[test_prompt, max_gen_length],
                outputs=generated_output
            )

            export_button.click(
                export_model_fn,
                inputs=[export_format],
                outputs=export_status
            )

        return app

if __name__ == '__main__':
    ui = GemmaUI()
    app = ui.create_ui()
    app.launch()