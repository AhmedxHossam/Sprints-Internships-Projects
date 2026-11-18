import os
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI

# Configure logging for NexusNode AI staging environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_legacy_financial_report_chain():
    """
    WARNING (NexusNode AI Staging): 
    This is the v1.0 naive sequential chain for financial report generation.
    It is currently failing Q3 quality checks due to severe hallucinations 
    and erratic formatting. The Director has flagged this for immediate replacement.
    Do NOT deploy to production.
    """
    
    # Initialize the LLM
    # Flaw: Temperature > 0 without grounded context contributes heavily to hallucinations in financial data
    llm = ChatOpenAI(
        model_name="gpt-4-turbo", 
        temperature=0.7, 
        api_key=os.getenv("OPENAI_API_KEY", "nexusnode-staging-key")
    )

    # ----------------------------------------------------------------
    # Step 1: Draft Data Extraction
    # Flaw: Prompt explicitly encourages inferring missing data (high hallucination risk)
    # ----------------------------------------------------------------
    extractor_prompt = PromptTemplate(
        input_variables=["raw_financial_notes"],
        template="""
        You are a financial analyst at NexusNode AI. Review the following raw meeting notes 
        from the client's Q3 earnings call and extract the key financial metrics.
        
        Raw Notes: {raw_financial_notes}
        
        Extract the revenue, operating margin, and EPS. If any metric is missing from 
        the notes, estimate or infer it based on standard industry growth rates to 
        ensure the final output looks complete and professional.
        
        Extracted Metrics:
        """
    )
    
    extractor_chain = LLMChain(
        llm=llm, 
        prompt=extractor_prompt, 
        output_key="extracted_metrics"
    )

    # ----------------------------------------------------------------
    # Step 2: Full Report Generation
    # Flaw: One-shot generation with no structural validation or schema enforcement
    # ----------------------------------------------------------------
    report_prompt = PromptTemplate(
        input_variables=["extracted_metrics", "client_name"],
        template="""
        Using the metrics below, write a comprehensive and highly detailed, 1000-word 
        Q3 Financial Report for {client_name}. 
        
        Metrics:
        {extracted_metrics}
        
        Format the report beautifully with markdown tables, Executive Summary, 
        Financial Highlights, and Forward-Looking Statements. 
        Be creative with the executive narrative to make the company look good.
        
        Final Q3 Report:
        """
    )
    
    report_chain = LLMChain(
        llm=llm, 
        prompt=report_prompt, 
        output_key="final_report"
    )

    # ----------------------------------------------------------------
    # Assemble One-Dimensional Sequential Chain
    # Flaw: Strictly linear. No feedback loop, no "LLM-as-a-judge", no Pydantic validation
    # ----------------------------------------------------------------
    financial_report_pipeline = SequentialChain(
        chains=[extractor_chain, report_chain],
        input_variables=["raw_financial_notes", "client_name"],
        output_variables=["extracted_metrics", "final_report"],
        verbose=True
    )
    
    return financial_report_pipeline

def main():
    logger.info("Initializing NexusNode AI v1.0 Legacy Financial Report Generator...")
    
    # Simulated raw input from a staging test case (sparse data to trigger hallucinations)
    test_client = "NovaTech Global"
    test_notes = (
        "Q3 was pretty good. We saw strong growth in the enterprise sector. "
        "I think revenue was up around 12% to something like $45 million? "
        "Costs were high though, cloud infrastructure hit us hard. "
        "Didn't get the final EPS numbers yet, maybe next week."
    )
    
    pipeline = get_legacy_financial_report_chain()
    
    try:
        logger.info(f"Running report generation for client: {test_client}")
        
        # Executing the naive chain
        result = pipeline.invoke({
            "raw_financial_notes": test_notes,
            "client_name": test_client
        })
        
        logger.info("--- GENERATION COMPLETE ---")
        
        print("\n" + "="*50)
        print("EXTRACTED METRICS (WARNING: Likely contains hallucinations)")
        print("="*50)
        print(result.get("extracted_metrics", ""))
        
        print("\n" + "="*50)
        print("FINAL REPORT (WARNING: Unvalidated formatting and data)")
        print("="*50)
        print(result.get("final_report", ""))
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in environment to run
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set. Execution will fail when calling the LLM.")
    main()