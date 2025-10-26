from transformers import pipeline
import torch
import json

pipe = pipeline(
    "text-generation",
    model="google/gemma-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    token="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",
    device="cuda",
)

with open('text.txt', 'r') as ques_file:
  questions = [line.strip() for line in ques_file]
  answer_array=[]
  for i in range(len(questions)):
    messages = [
      {"role": "user", "content": "questions[i]"},
    ]
    outputs = pipe(
      messages,
      max_new_tokens=256,
      do_sample=True,
      temperature=0.7,
      top_k=50,
      top_p=0.95
  )
  assistant_response = outputs[0]["generated_text"][-1]["content"]
  answer_array.insert(i,assistant_response)
with open('answer.json', 'w') as answer_file:
  json.dump(answer_array,answer_file)
