This folder contains the source code for TusoAI.

construct_categories.py contains code for generating and refining instruction categories.
construct_initializations.py contains code for generating and refining initial solution descriptions.
construct_prompts.py contains code for generating and refining instructions within categories.
diagnostic_prompts.json contains a list of the default diagnostic prompts.
extract_literature.py contains code for extracting and parsing literature from Semantic Scholar.
generic_prompts.json contains some general prompts, used in the ablation studies.
initialize_llm.py contains code for generic LLM client initializations.
optimizer.py contains much of the TusoAI method, including generating initial solutions, refining them, debugging, and returning the finally optimized code.
summarize_literature.py contains code for summarizing the literature extracted from Semantic Scholar into short technical summaries.
tusoai.py is a general class that organizes calls to other files.
