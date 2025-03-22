import gradio as gr
import threading
import queue
from utils.check_dataset import validate_dataset, generate_dataset_report
from utils.sample_dataset import generate_sample_datasets
from utils.model import GemmaFineTuning
from utils.logging import LogManager

class GemmaUI:
    def __init__(self):
        self.model_instance = GemmaFineTuning()
        self.default_params = self.model_instance.default_params
        self.log_queue = queue.Queue()
        self.training_active = False
        self.log_refresh_interval = 1.0  # seconds
        self.training_progress = {
            "total_steps": 0,
            "current_step": 0,
            "current_loss": 0.0,
            "avg_loss": 0.0,
            "progress_percentage": 0.0
        }
        self.log_manager = LogManager()

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
                                    "unsloth/gemma-2b-it-4bit",
                                    "unsloth/gemma-7b-it-4bit",
                                    "unsloth/gemma-2b-4bit",
                                    "unsloth/gemma-7b-4bit"
                                ],
                                value="unsloth/gemma-2b-it-4bit",
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
                        with gr.Column(scale=1):
                            start_training_button = gr.Button("Start Fine-tuning", variant="primary")
                            stop_training_button = gr.Button("Stop Training", variant="stop")
                            training_status = gr.Textbox(label="Training Status", interactive=False)
                            
                            # Add detailed logging components
                            with gr.Accordion("Training Logs", open=True):
                                log_output = gr.Markdown("")
                                auto_scroll = gr.Checkbox(value=True, label="Auto-scroll logs")
                                clear_logs = gr.Button("Clear Logs")

                        with gr.Column(scale=1):
                            progress_plot = gr.Plot(label="Training Progress")
                            refresh_plot_button = gr.Button("Refresh Plot")
                            
                            # Add training metrics
                            with gr.Accordion("Training Metrics", open=True):
                                current_loss = gr.Number(label="Current Loss", value=0.0)
                                avg_loss = gr.Number(label="Average Loss", value=0.0)
                                progress_bar = gr.Slider(
                                    label="Training Progress",
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    interactive=False
                                )
                            
                            # Add terminal-like log viewer
                            with gr.Accordion("Training Logs", open=True):
                                terminal = gr.Code(
                                    label="Log Terminal",
                                    language="bash",
                                    interactive=False,
                                    lines=20,
                                    value="=== Training Log ===\n"
                                )
                                log_file_path = gr.Textbox(
                                    label="Log File Path",
                                    value=self.log_manager.get_current_log_file(),
                                    interactive=False
                                )
                                with gr.Row():
                                    auto_scroll = gr.Checkbox(value=True, label="Auto-scroll logs")
                                    clear_logs = gr.Button("Clear")
                                    refresh_logs = gr.Button("Refresh")
                                    open_logs = gr.Button("Open Log File")

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

            def update_logs():
                """Update logs from queue"""
                logs = []
                try:
                    while True:
                        log = self.log_queue.get_nowait()
                        logs.append(log)
                except queue.Empty:
                    pass
                
                if logs:
                    log_text = "\n".join(logs)
                    return log_text

            def add_log(message):
                """Add log message to queue and terminal"""
                self.log_queue.put(f"[LOG] {message}")
                self.log_manager.log(message)
                return update_logs()

            def clear_log_output():
                """Clear the log output"""
                while not self.log_queue.empty():
                    self.log_queue.get()
                return ""

            def preprocess_data(file, format_type):
                try:
                    if file is None:
                        return "Please upload a file first.", update_logs()
                    
                    add_log(f"Processing {file.name} as {format_type} format...")
                    
                    # Update validate_dataset call without self
                    validation_results = validate_dataset(file.name, format_type)
                    validation_report = generate_dataset_report(validation_results)
                    add_log(validation_report)
                    
                    if not validation_results["is_valid"]:
                        return "Dataset validation failed. See logs for details.", update_logs()

                    # Process the dataset
                    dataset = self.model_instance.prepare_dataset(file.name, format_type)
                    self.model_instance.dataset = dataset
                    
                    num_samples = len(dataset["train"])
                    add_log(f"Successfully loaded {num_samples} training examples")
                    
                    # Sample examples
                    examples = dataset["train"].select(range(min(3, num_samples)))
                    for i, ex in enumerate(examples):
                        add_log(f"\nExample {i+1}:")
                        add_log(f"{ex['text'][:200]}...")
                    
                    return "Dataset processed successfully!", update_logs()
                except Exception as e:
                    add_log(f"Error: {str(e)}")
                    return f"Error preprocessing data: {str(e)}", update_logs()

            def start_training(*params):
                try:
                    if self.training_active:
                        return "Training is already in progress.", update_logs()
                    
                    self.training_active = True
                    add_log("Initializing training...")

                    [model_name, learning_rate, batch_size, epochs, max_length,
                     use_lora, lora_r, lora_alpha, eval_ratio] = params
                     
                    add_log("Step 1: Downloading and loading model...")
                    # Prepare training parameters
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
                    
                    add_log("Step 2: Model parameters:")
                    for k, v in training_params.items():
                        add_log(f"  {k}: {v}")
                    
                    add_log("Step 3: Preparing dataset (if needed)...")
                    # (Assuming dataset was already preprocessed via preprocess_data)
                    
                    def train_thread():
                        try:
                            add_log("Step 4: Starting fine-tuning...")
                            status = self.model_instance.train(training_params)
                            add_log(status)
                        except Exception as e:
                            add_log(f"Training error: {str(e)}")
                        finally:
                            self.training_active = False
                    threading.Thread(target=train_thread, daemon=True).start()
                    
                    return "Training started! Monitor the progress in the logs.", update_logs()
                    
                except Exception as e:
                    self.training_active = False
                    add_log(f"Error starting training: {str(e)}")
                    return f"Error starting training: {str(e)}", update_logs()

            def stop_training():
                if self.model_instance.trainer is not None:
                    # Attempt to stop the trainer
                    self.model_instance.trainer.stop_training = True
                    add_log("Training stop signal sent. It may take a moment to complete the current step.")
                    return "Training stop signal sent. It may take a moment to complete the current step.", update_logs()
                return "No active training to stop.", update_logs()

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

            def refresh_terminal():
                """Refresh terminal with latest logs"""
                return self.log_manager.read_logs()

            def open_log_file():
                """Open log file in default text editor"""
                import webbrowser
                webbrowser.open(self.log_manager.get_current_log_file())

            # Connect UI components to functions
            preprocess_button.click(
                preprocess_data,
                inputs=[file_upload, file_format],
                outputs=[dataset_info, log_output]
            )

            start_training_button.click(
                start_training,
                inputs=[
                    model_name, learning_rate, batch_size, epochs, max_length,
                    use_lora, lora_r, lora_alpha, eval_ratio
                ],
                outputs=[training_status, log_output]
            )

            stop_training_button.click(
                stop_training,
                inputs=[],
                outputs=[training_status, log_output]
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

            clear_logs.click(
                clear_log_output,
                outputs=[log_output]
            )

            refresh_logs.click(
                refresh_terminal,
                outputs=[terminal]
            )

            open_logs.click(
                fn=open_log_file
            )

            # Replace the gr.on() with proper event handling
            def periodic_log_update():
                """Periodic log update function"""
                logs = update_logs()
                if logs:
                    return logs
                return gr.update()

            # Set up automatic log refresh
            refresh_event = gr.on(
                triggers=[
                    start_training_button.click,
                    stop_training_button.click,
                    preprocess_button.click
                ],
                fn=periodic_log_update,
                outputs=log_output
            ).then(
                fn=lambda: gr.update(interval=self.log_refresh_interval),
                outputs=log_output
            )

            # Add manual refresh button for logs
            log_refresh_button = gr.Button("Refresh Logs")
            log_refresh_button.click(
                fn=update_logs,
                outputs=log_output
            )

            def update_progress():
                """Update training progress indicators"""
                try:
                    progress_data = {
                        progress_bar: self.training_progress["progress_percentage"],
                        current_loss: self.training_progress["current_loss"],
                        avg_loss: self.training_progress["avg_loss"],
                        training_status: f"Step {self.training_progress['current_step']}/{self.training_progress['total_steps']}",
                        terminal: self.log_manager.read_logs() if auto_scroll.value else gr.skip()
                    }
                    return progress_data
                except Exception as e:
                    return None

            # Add progress update interval
            progress_update = gr.on(
                triggers=start_training_button.click,
                fn=update_progress,
                outputs=[progress_bar, current_loss, avg_loss, training_status],
                stream_every=1  # Update every second
            )

        return app

if __name__ == '__main__':
    ui = GemmaUI()
    app = ui.create_ui()
    # Fix the queue configuration
    app.queue()  # Enable queuing for async updates
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        max_threads=3  # Control concurrent operations
    )