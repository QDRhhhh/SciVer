from string import Template

MAX_TOKENS = 10240
GENERATION_TEMPERATURE = 1.0
GENERATION_SEED = 215



OPEN_ENDED_COT_PROMPT1 = Template("""
Claim: $claim
Context: $context
Caption: $caption

Your task is to critically evaluate the claim based on the image and the caption. Carefully examine whether the information in the caption truly supports the claim. Be skeptical and cautious: if there is any inconsistency, missing evidence, or ambiguity, consider the claim incorrect.

Start by explaining your reasoning process clearly, focusing on identifying potential contradictions, lack of support, or misleading interpretations. If the claim is unsupported or contradicted by the caption and image, respond with 'no'. Only respond with 'yes' if the claim is fully and clearly supported.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

OPEN_ENDED_COT_PROMPT2 = Template("""
Claim: $claim
Context: $context
item1 Caption: $caption1
item2 Caption: $caption2

You will be provided with two images, two tables, or one image and one table, along with contextual information explaining their relationship. Your task is to critically assess whether the claim is fully and clearly supported by the provided evidence.

Approach the task with skepticism. Carefully examine both the captions and the context for contradictions, missing links, or insufficient support. If there is any ambiguity, inconsistency, or lack of direct evidence, consider the claim incorrect. Only answer 'yes' if the claim is explicitly and unambiguously confirmed by the content.

Begin by explaining your reasoning step by step, focusing on identifying potential errors or unsupported assumptions. Conclude by stating your judgment using the format: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer.
""")

OPEN_ENDED_COT_PROMPT3 = Template("""
Claim: $claim
Context: $context
Caption: $caption

Your task is to evaluate the claim based on the image and the caption. Carefully examine whether the information in the caption truly supports the claim. Apply any relevant scientific principles, statistical logic, or domain knowledge necessary to link the evidence to the claim. Be balanced: actively look for confirming details as well as inconsistencies, missing evidence, or ambiguities.

Start by explaining your reasoning process step by step—describe what you observe in the image and caption, what background knowledge you use, and how you test whether each key part of the claim is supported. If every essential component of the claim is clearly and completely backed by the caption and image, respond with 'yes'. If any critical point is contradicted, unsupported, or unclear, respond with 'no'.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

OPEN_ENDED_COT_PROMPT4 = Template("""
Claim: $claim
Context: $context
item1 Caption: $caption1
item2 Caption: $caption2

You will be provided with two images, two tables, or one image and one table, along with contextual information explaining their relationship. Your task is to critically assess whether the claim is fully and clearly supported by the provided evidence.

Approach the task with skepticism. Carefully examine both the captions and the context for contradictions, missing links, or insufficient support. If there is any ambiguity, inconsistency, or lack of direct evidence, consider the claim incorrect. Only answer 'yes' if the claim is explicitly and unambiguously confirmed by the content.

Begin by explaining your reasoning step by step, focusing on identifying potential errors or unsupported assumptions. Conclude by stating your judgment using the format: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer.
""")


COT_PROMPT = {
    "direct": OPEN_ENDED_COT_PROMPT1,
    "analytical":OPEN_ENDED_COT_PROMPT3,
    "parallel":OPEN_ENDED_COT_PROMPT2,
    "sequential":OPEN_ENDED_COT_PROMPT4
}