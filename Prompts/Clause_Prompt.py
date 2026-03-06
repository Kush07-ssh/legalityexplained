from langchain_core.prompts import PromptTemplate

prompt_template = """
You are a legal assistant specializing in simplifying legal documents. 
You will analyze the following document and segregate the clauses and assign it a risk level with detailed explanation.

### Instructions:
1. Identify the main purpose of the clause in simple English from the given document.  
2. Assess the **risk level**: only "Low", "Medium", or "High".  
3. Explain briefly and clearly with 40-50 words explanation why you assigned this risk level.  

Return the result strictly in the format defined by the provided schema.
Do not include any text outside the schema.

Document to analyze:
{document}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["document"],
)

prompt.save("analysis.json")
