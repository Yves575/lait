import argparse
import pandas as pd
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM evaluation using vLLM')
    parser.add_argument('--file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--test_size', type=int, default=None, help='Number of samples to test')
    parser.add_argument('--model_name', type=str, default="Unbabel/M-Prometheus-14B", 
                        help='HuggingFace model name')
    parser.add_argument('--cache_dir', type=str, default="./cache", 
                        help='Directory to cache models')
    parser.add_argument('--output_file', type=str, default="m-premetheous_results.csv", 
                        help='Output file to save results')
    parser.add_argument('--device', type=str, default="cuda:1", help='Device to run the model on')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for the model')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Maximum tokens to generate')
    return parser.parse_args()

def retrive_prompt(df):
    messages = []
    for idx, row in df.iterrows():
        instruction_ = row["source"]
        response_ = row["tgt"]
        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
        rubric_data = {
      "criteria":"How good is the translation quality of literary source texts?",
      "score0_description":"Nonsense/No meaning preserved: Nearly all information is lost between the translation and source. Grammar is irrelevant.",
      "score2_description":"Some Meaning Preserved: The translation preserves some of the meaning of the source but misses significant parts. The narrative is hard to follow due to fundamental errors. Grammar may be poor.",
      "score4_description":"Most Meaning Preserved and Few Grammar Mistakes: The translation retains most of the meaning of the source. This may contain some grammar mistakes or minor contextual inconsistencies.",
      "score6_description":"Perfect Meaning and Grammar: The meaning of the translation is completely consistent with the source and the surrounding context (if applicable). The grammar is also correct.",
    }
        ABSOLUTE_PROMPT = """###Task Description:
            An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
            1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
            2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
            3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
            4. Please do not generate any other opening, closing, and explanations.
            
            ###The instruction to evaluate (the source texts):
            {instruction}
            
            ###Response to evaluate:
            {response}
            
            ###Score Rubrics:
            {rubric}
            
            ###Feedback: """
    
        user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(instruction = instruction_, response = response_, rubric = rubric_data) # Fill the prompt with your data
    
        m = [{"role": "user", "content": user_content},]
        messages.append(m)
    return messages

def extract_score_from_response(response):
    try:
        # Look for the score pattern in the response
        score_part = response.split("[RESULT]")[-1].strip()
        score = int(score_part.split()[0])
        return min(max(score, 1), 5)  # Ensure score is between 1 and 5
    except:
        return None

def main():
    args = parse_args()
    
    # Load the CSV file
    df = pd.read_csv(args.file)
    if args.test_size:
        df = df.head(args.test_size)
    
    # Initialize vLLM
    llm = LLM(
        model=args.model_name,
        download_dir=args.cache_dir,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9
    )
    
    # Get prompts
    prompts = retrive_prompt(df)
    inputs = [llm.llm_engine.tokenizer.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
          for prompt in prompts]
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_model_len=15000,
        max_tokens=args.max_tokens
    )
    
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Process results
    results = []
    for output in outputs:
        response = output.outputs[0].text
        score = extract_score_from_response(response)
        results.append({
            'response': response,
            'score': score
        })
    
    # Add results to dataframe
    df['llm_response'] = [r['response'] for r in results]
    df['llm_score'] = [r['score'] for r in results]
    
    # Save results
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()



