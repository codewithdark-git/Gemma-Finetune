def validate_dataset(file_path, format_type):  # Remove self parameter
    """
    Validate and analyze the dataset format, providing detailed feedback

    Parameters:
    file_path (str): Path to the dataset file
    format_type (str): File format (csv, jsonl, text)

    Returns:
    dict: Validation results including format, structure, and statistics
    """
    import pandas as pd
    import json
    import os
    import re

    validation_results = {
        "is_valid": False,
        "format": format_type,
        "detected_structure": None,
        "statistics": {},
        "issues": [],
        "recommendations": []
    }

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            validation_results["issues"].append(f"File not found: {file_path}")
            return validation_results

        # Check file size
        file_size = os.path.getsize(file_path)
        validation_results["statistics"]["file_size_bytes"] = file_size
        validation_results["statistics"]["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        if file_size == 0:
            validation_results["issues"].append("File is empty")
            return validation_results

        if format_type == "csv":
            # Load CSV file
            try:
                df = pd.read_csv(file_path)
                validation_results["statistics"]["total_rows"] = len(df)
                validation_results["statistics"]["total_columns"] = len(df.columns)
                validation_results["statistics"]["column_names"] = list(df.columns)

                # Check for null values
                null_counts = df.isnull().sum().to_dict()
                validation_results["statistics"]["null_counts"] = null_counts

                if validation_results["statistics"]["total_rows"] == 0:
                    validation_results["issues"].append("CSV file has no rows")
                    return validation_results

                # Detect structure
                if "instruction" in df.columns and "response" in df.columns:
                    validation_results["detected_structure"] = "instruction-response"
                    validation_results["is_valid"] = True
                elif "input" in df.columns and "output" in df.columns:
                    validation_results["detected_structure"] = "input-output"
                    validation_results["is_valid"] = True
                elif "prompt" in df.columns and "completion" in df.columns:
                    validation_results["detected_structure"] = "prompt-completion"
                    validation_results["is_valid"] = True
                elif "text" in df.columns:
                    validation_results["detected_structure"] = "text-only"
                    validation_results["is_valid"] = True
                else:
                    # Look for text columns
                    text_columns = [col for col in df.columns if df[col].dtype == 'object']
                    if text_columns:
                        validation_results["detected_structure"] = "custom"
                        validation_results["statistics"]["potential_text_columns"] = text_columns
                        validation_results["is_valid"] = True
                        validation_results["recommendations"].append(
                            f"Consider renaming columns to match standard formats: instruction/response, input/output, prompt/completion, or text"
                        )
                    else:
                        validation_results["issues"].append("No suitable text columns found in CSV")

                # Check for short text
                if validation_results["detected_structure"] == "instruction-response":
                    short_instructions = (df["instruction"].str.len() < 10).sum()
                    short_responses = (df["response"].str.len() < 10).sum()
                    validation_results["statistics"]["short_instructions"] = short_instructions
                    validation_results["statistics"]["short_responses"] = short_responses

                    if short_instructions > 0:
                        validation_results["issues"].append(f"Found {short_instructions} instructions shorter than 10 characters")
                    if short_responses > 0:
                        validation_results["issues"].append(f"Found {short_responses} responses shorter than 10 characters")

            except Exception as e:
                validation_results["issues"].append(f"Error parsing CSV: {str(e)}")
                return validation_results

        elif format_type == "jsonl":
            try:
                # Load JSONL file
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json_obj = json.loads(line)
                            data.append(json_obj)
                        except json.JSONDecodeError:
                            validation_results["issues"].append(f"Invalid JSON at line {line_num}")

                validation_results["statistics"]["total_examples"] = len(data)

                if len(data) == 0:
                    validation_results["issues"].append("No valid JSON objects found in file")
                    return validation_results

                # Get sample of keys from first object
                if data:
                    validation_results["statistics"]["sample_keys"] = list(data[0].keys())

                # Detect structure
                structures = []
                for item in data:
                    if "instruction" in item and "response" in item:
                        structures.append("instruction-response")
                    elif "input" in item and "output" in item:
                        structures.append("input-output")
                    elif "prompt" in item and "completion" in item:
                        structures.append("prompt-completion")
                    elif "text" in item:
                        structures.append("text-only")
                    else:
                        structures.append("custom")

                # Count structure types
                from collections import Counter
                structure_counts = Counter(structures)
                validation_results["statistics"]["structure_counts"] = structure_counts

                # Set detected structure to most common
                if structures:
                    most_common = structure_counts.most_common(1)[0][0]
                    validation_results["detected_structure"] = most_common
                    validation_results["is_valid"] = True

                    # Check if mixed
                    if len(structure_counts) > 1:
                        validation_results["issues"].append(f"Mixed structures detected: {dict(structure_counts)}")
                        validation_results["recommendations"].append("Consider standardizing all records to the same structure")

            except Exception as e:
                validation_results["issues"].append(f"Error parsing JSONL: {str(e)}")
                return validation_results

        elif format_type == "text":
            try:
                # Read text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Get basic stats
                total_chars = len(content)
                total_words = len(content.split())
                total_lines = content.count('\n') + 1

                validation_results["statistics"]["total_characters"] = total_chars
                validation_results["statistics"]["total_words"] = total_words
                validation_results["statistics"]["total_lines"] = total_lines

                # Check if it's a single large document or multiple examples
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                validation_results["statistics"]["total_paragraphs"] = len(paragraphs)

                # Try to detect structure
                # Look for common patterns like "Q: ... A: ...", "Input: ... Output: ..."
                has_qa_pattern = re.search(r"Q:.*?A:", content, re.DOTALL) is not None
                has_input_output = re.search(r"Input:.*?Output:", content, re.DOTALL) is not None
                has_prompt_completion = re.search(r"Prompt:.*?Completion:", content, re.DOTALL) is not None

                if has_qa_pattern:
                    validation_results["detected_structure"] = "Q&A-format"
                elif has_input_output:
                    validation_results["detected_structure"] = "input-output-format"
                elif has_prompt_completion:
                    validation_results["detected_structure"] = "prompt-completion-format"
                elif len(paragraphs) > 1:
                    validation_results["detected_structure"] = "paragraphs"
                else:
                    validation_results["detected_structure"] = "continuous-text"

                validation_results["is_valid"] = True

                if validation_results["detected_structure"] == "continuous-text" and total_chars < 1000:
                    validation_results["issues"].append("Text file is very short for fine-tuning")
                    validation_results["recommendations"].append("Consider adding more content or examples")

            except Exception as e:
                validation_results["issues"].append(f"Error parsing text file: {str(e)}")
                return validation_results
        else:
            validation_results["issues"].append(f"Unsupported file format: {format_type}")
            return validation_results

        # General recommendations
        if validation_results["is_valid"]:
            if not validation_results["issues"]:
                validation_results["recommendations"].append("Dataset looks good and ready for fine-tuning!")
            else:
                validation_results["recommendations"].append("Address the issues above before proceeding with fine-tuning")

        return validation_results

    except Exception as e:
        validation_results["issues"].append(f"Unexpected error: {str(e)}")
        return validation_results

def generate_dataset_report(validation_results):
    """
    Generate a user-friendly report from validation results

    Parameters:
    validation_results (dict): Results from validate_dataset

    Returns:
    str: Formatted report
    """
    report = []

    # Add header
    report.append("# Dataset Validation Report")
    report.append("")

    # Add validation status
    if validation_results["is_valid"]:
        report.append("✅ Dataset is valid and can be used for fine-tuning")
    else:
        report.append("❌ Dataset has issues that need to be addressed")
    report.append("")

    # Add format info
    report.append(f"**File Format:** {validation_results['format']}")
    report.append(f"**Detected Structure:** {validation_results['detected_structure']}")
    report.append("")

    # Add statistics
    report.append("## Statistics")
    for key, value in validation_results["statistics"].items():
        # Format the key for better readability
        formatted_key = key.replace("_", " ").title()
        report.append(f"- {formatted_key}: {value}")
    report.append("")

    # Add issues
    if validation_results["issues"]:
        report.append("## Issues")
        for issue in validation_results["issues"]:
            report.append(f"- ⚠️ {issue}")
        report.append("")

    # Add recommendations
    if validation_results["recommendations"]:
        report.append("## Recommendations")
        for recommendation in validation_results["recommendations"]:
            report.append(f"- 💡 {recommendation}")

    return "\n".join(report)