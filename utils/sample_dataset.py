import pandas as pd
import json
import os
import random

def generate_sample_datasets(output_dir="./sample_datasets"):
    """
    Generate sample datasets in CSV, JSONL, and text formats for testing the Gemma fine-tuning UI

    Parameters:
    output_dir (str): Directory to save the sample datasets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Sample data for instruction-response pairs
    instruction_data = [
        {"instruction": "Explain the concept of machine learning in simple terms.",
         "response": "Machine learning is when computers learn from examples rather than being explicitly programmed. It's like teaching a child by showing examples instead of giving exact rules."},
        {"instruction": "What are the health benefits of regular exercise?",
         "response": "Regular exercise improves cardiovascular health, strengthens muscles and bones, helps control weight, reduces stress and anxiety, improves sleep quality, and increases overall energy levels."},
        {"instruction": "How do I make a basic pasta sauce?",
         "response": "For a basic tomato pasta sauce, sauté minced garlic and onions in olive oil until soft. Add canned crushed tomatoes, salt, pepper, and Italian herbs. Simmer for 20-30 minutes, stirring occasionally. Add basil at the end for fresh flavor."},
        {"instruction": "Summarize the water cycle.",
         "response": "The water cycle is the continuous movement of water on, above, and below Earth's surface. It involves evaporation (water turning to vapor), condensation (vapor forming clouds), precipitation (rain or snow), and collection (water returning to bodies of water)."},
        {"instruction": "What is the difference between a simile and a metaphor?",
         "response": "A simile compares two things using 'like' or 'as' (e.g., 'as brave as a lion'). A metaphor directly states that one thing is another (e.g., 'he is a lion in battle'). Both are figurative language techniques used to create vivid descriptions."}
    ]

    # Generate more instruction-response pairs
    topics = ["history", "science", "literature", "cooking", "technology", "health", "travel", "sports", "music", "art"]
    question_starters = ["Explain", "Describe", "How to", "What is", "Why does", "Compare", "Summarize", "List ways to", "Define", "Analyze"]

    for _ in range(20):
        topic = random.choice(topics)
        starter = random.choice(question_starters)
        instruction = f"{starter} {topic.lower()} {random.choice(['concepts', 'principles', 'ideas', 'techniques', 'methods'])}"
        response = f"This is a sample response about {topic} that would be more detailed in a real dataset. It would contain multiple sentences explaining {topic} concepts in depth."
        instruction_data.append({"instruction": instruction, "response": response})

    # Create a dictionary to store sample datasets
    datasets = {}

    # 1. Create CSV in instruction-response format
    df_instruction = pd.DataFrame(instruction_data)
    datasets["instruction_response.csv"] = df_instruction

    # 2. Create CSV in input-output format
    input_output_data = [{"input": item["instruction"], "output": item["response"]} for item in instruction_data]
    df_input_output = pd.DataFrame(input_output_data)
    datasets["input_output.csv"] = df_input_output

    # 3. Create CSV in text-only format
    text_data = [{"text": f"Q: {item['instruction']}\nA: {item['response']}"} for item in instruction_data]
    df_text = pd.DataFrame(text_data)
    datasets["text_only.csv"] = df_text

    # 4. Create CSV with non-standard format
    custom_data = [{"question": item["instruction"], "answer": item["response"]} for item in instruction_data]
    df_custom = pd.DataFrame(custom_data)
    datasets["custom_format.csv"] = df_custom

    # 5. Create JSONL in instruction-response format
    jsonl_instruction = instruction_data
    datasets["instruction_response.jsonl"] = jsonl_instruction

    # 6. Create JSONL in prompt-completion format
    prompt_completion_data = [{"prompt": item["instruction"], "completion": item["response"]} for item in instruction_data]
    datasets["prompt_completion.jsonl"] = prompt_completion_data

    # 7. Create JSONL with non-standard format
    jsonl_custom = [{"query": item["instruction"], "result": item["response"]} for item in instruction_data]
    datasets["custom_format.jsonl"] = jsonl_custom

    # 8. Create text format (paragraphs)
    text_paragraphs = "\n\n".join([f"Q: {item['instruction']}\nA: {item['response']}" for item in instruction_data])
    datasets["paragraphs.txt"] = text_paragraphs

    # 9. Create text format (single examples per line)
    text_lines = "\n".join([f"{item['instruction']} => {item['response']}" for item in instruction_data])
    datasets["single_lines.txt"] = text_lines

    # Save all datasets
    for filename, data in datasets.items():
        file_path = os.path.join(output_dir, filename)

        if filename.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif filename.endswith('.jsonl'):
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
        elif filename.endswith('.txt'):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)

    print(f"Sample datasets generated in {output_dir}")
    return list(datasets.keys())

# if __name__ == "__main__":
#     # Generate sample datasets
#     generated_files = generate_sample_datasets()
#     print(f"Generated {len(generated_files)} sample dataset files:")
#     for file in generated_files:
#         print(f" - {file}")