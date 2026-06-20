import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Rate limiting for Gemini Free Tier
# Free tier: 15 requests per minute (RPM), 1M tokens per day
class RateLimiter:
    def __init__(self, max_requests_per_minute=15):
        self.max_requests = max_requests_per_minute
        self.requests = []
    
    def wait_if_needed(self):
        """Wait if we've hit the rate limit (15 RPM for free tier)"""
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(minutes=1)]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]).total_seconds() + 1
            if sleep_time > 0:
                print(f"⏳ Rate limit approached. Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.requests = []
        
        self.requests.append(datetime.now())

rate_limiter = RateLimiter(max_requests_per_minute=15)

# Import retrieval functions
# Ensure these exist in your 'retrieval.py' file or comment them out if testing locally without them
try:
    from retrieval import hybrid_search, keyword_search, semantic_search
except ImportError:
    print("⚠ Warning: 'retrieval' module not found. Mocking search functions for testing.")
    def hybrid_search(*args, **kwargs): return []
    def keyword_search(*args, **kwargs): return []
    def semantic_search(*args, **kwargs): return []

# Load environment variables
load_dotenv()

# Configure Gemini API with validation
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None


def get_gemini_client():
    """Return a Gemini client, initializing it lazily after .env is loaded."""
    global client
    if client:
        return client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    return client


def validate_gemini_api_key():
    """Fail fast when the app starts without Gemini credentials."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set. Add it to your .env file.")

# Define RAG prompt template
RAG_PROMPT_TEMPLATE = """You are an expert AI assistant specialized in Retrieval-Augmented Generation (RAG) and Large Language Models.

Use the following retrieved documents to answer the user's question accurately and comprehensively.
If the retrieved documents contain relevant information, use it to provide a well-structured answer.
If the documents don't contain relevant information, explicitly state that you don't have enough information.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Be accurate and cite the source documents when relevant
- Structure your answer clearly with sections if needed
- Be concise but comprehensive
- If information is incomplete, acknowledge it

YOUR ANSWER:
"""


def format_retrieved_contexts(results):
    """Format OpenSearch hits as model-readable context blocks."""
    contexts = []
    for i, hit in enumerate(results, 1):
        source = hit.get("_source", {})
        content = source.get("content", "")
        content_type = source.get("content_type", "text")
        file_name = source.get("file_name", "unknown")
        page_number = source.get("page_number")
        chunk_id = source.get("chunk_id") or hit.get("_id", "unknown")
        score = hit.get("_score", 0)
        page_label = page_number if page_number is not None else "unknown"

        context_entry = f"""
[Document {i}]
Type: {content_type}
File: {file_name}
Page: {page_label}
Chunk ID: {chunk_id}
Relevance Score: {score:.4f}
Content:
{content}
"""
        contexts.append(context_entry)

    return "\n---\n".join(contexts)


def format_source_citations(results):
    """Format page-level source citations for display below an answer."""
    if not results:
        return ""

    citation_lines = ["\n\nSources:"]
    for i, hit in enumerate(results, 1):
        source = hit.get("_source", {})
        file_name = source.get("file_name", "unknown")
        page_number = source.get("page_number")
        chunk_id = source.get("chunk_id") or hit.get("_id", "unknown")
        score = hit.get("_score", 0)
        page_label = page_number if page_number is not None else "unknown"
        short_chunk_id = str(chunk_id)[:12]
        citation_lines.append(
            f"- [{i}] {file_name}, page {page_label}, chunk `{short_chunk_id}`, score {score:.4f}"
        )

    return "\n".join(citation_lines)


def build_rag_prompt(context_text, question):
    """Build the final prompt sent to Gemini."""
    return RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)


def generate_with_gemini_streaming(prompt_text, model_name="gemini-2.0-flash"):
    """
    Generate response using Google's Gemini API with streaming.
    Respects Gemini Free Tier rate limits (15 RPM, 1M tokens/day).
    
    Args:
        prompt_text (str): The formatted prompt text
        model_name (str): Gemini model to use
    
    Returns:
        generator: Streaming text chunks
    """
    gemini_client = get_gemini_client()
    if not gemini_client:
        yield "Error: Client not initialized."
        return

    try:
        # Apply rate limiting for free tier
        rate_limiter.wait_if_needed()
        
        print("✓ Streaming response generation started...")
        
        # Create prompt part
        prompt_part = types.Part.from_text(text=prompt_text)

        # FIX: Use generate_content_stream and remove stream=True argument
        response_stream = gemini_client.models.generate_content_stream(
            model=model_name,
            contents=[prompt_part],
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            )
        )

        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
                        
    except Exception as e:
        error_msg = f"✗ Error with Gemini streaming: {str(e)}"
        print(error_msg)
        yield error_msg


def generate_with_gemini(prompt_text, model_name="gemini-2.0-flash"):
    """
    Generate response using Google's Gemini API (non-streaming).
    Respects Gemini Free Tier rate limits (15 RPM, 1M tokens/day).
    
    Args:
        prompt_text (str): The formatted prompt text
        model_name (str): Gemini model to use
    
    Returns:
        str: Response text
    """
    gemini_client = get_gemini_client()
    if not gemini_client:
        return "Error: Client not initialized."

    try:
        # Apply rate limiting for free tier
        rate_limiter.wait_if_needed()
        
        print("✓ Requesting non-streaming response...")

        # Validate prompt length
        if len(prompt_text) > 100000:
            print(f"⚠ Warning: Prompt is {len(prompt_text)} characters. Truncating to 100000...")
            prompt_text = prompt_text[:100000] + "\n\n[Content truncated due to length]"

        # Create prompt part
        prompt_part = types.Part.from_text(text=prompt_text)

        response = gemini_client.models.generate_content(
            model=model_name,
            contents=[prompt_part],
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            ),
        )

        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "⚠ No response generated. The model may have blocked the request."

    except Exception as e:
        error_msg = f"✗ Error with Gemini generation: {str(e)}"
        print(error_msg)
        return error_msg


def generate_rag_response(
    query, search_type="hybrid", top_k=5, stream=False, model_name="gemini-2.0-flash"
):
    """
    Generate RAG response using retrieved chunks from OpenSearch.

    Args:
        query (str): User's question
        search_type (str): Type of search - 'keyword', 'semantic', or 'hybrid' (default)
        top_k (int): Number of documents to retrieve (default: 5)
        stream (bool): Whether to stream the response (default: False)
        model_name (str): Gemini model to use

    Returns:
        str or generator: Generated response or streaming generator
    """
    try:
        if not query or not query.strip():
            return "⚠ Please enter a question."

        search_type = search_type if search_type in {"keyword", "semantic", "hybrid"} else "hybrid"
        try:
            top_k = max(1, min(int(top_k), 10))
        except (TypeError, ValueError):
            top_k = 5

        print(f"\n{'='*80}")
        print(f"RAG GENERATION PIPELINE")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Search Type: {search_type}")
        print(f"Top K: {top_k}")
        print(f"Model: {model_name}")
        print(f"Streaming: {stream}\n")

        # Step 1: Retrieve relevant documents
        print("Step 1: Retrieving relevant documents...")
        if search_type == "keyword":
            results = keyword_search(query, top_k=top_k)
        elif search_type == "semantic":
            results = semantic_search(query, top_k=top_k)
        else:  # hybrid (default)
            results = hybrid_search(query, top_k=top_k)

        if not results:
            message = "⚠ No relevant documents found. Please try a different search type or refine your question."
            print(message)
            return message

        print(f"✓ Retrieved {len(results)} documents\n")
        citations_text = format_source_citations(results)

        # Step 2: Format retrieved contexts
        print("Step 2: Formatting retrieved contexts...")
        context_text = format_retrieved_contexts(results)
        print(f"✓ Formatted {len(results)} contexts\n")

        # Step 4: Format the prompt
        print("Step 3: Formatting prompt for Gemini...")
        prompt_text = build_rag_prompt(context_text, query)
        print(f"✓ Prompt size: {len(prompt_text)} characters\n")

        # Step 5: Generate response with Gemini
        print("Step 4: Generating response with Gemini...\n")
        print("="*80)
        
        if stream:
            def stream_with_citations():
                for chunk in generate_with_gemini_streaming(prompt_text, model_name=model_name):
                    yield chunk
                if citations_text:
                    yield citations_text

            return stream_with_citations()
        else:
            result = generate_with_gemini(prompt_text, model_name=model_name)
            print("="*80)
            return f"{result}{citations_text}" if citations_text else result

    except Exception as e:
        error_message = f"✗ Error in RAG process: {str(e)}"
        print(error_message)
        return error_message


def interactive_rag():
    """
    Interactive RAG chat interface.
    """
    print("\n" + "="*80)
    print("INTERACTIVE RAG CHATBOT")
    print("="*80)
    print("\nCommands:")
    print("  'exit' or 'quit' - Exit the chatbot")
    print("  'help' - Show help information")
    print("\nSearch types: keyword, semantic, hybrid (default)\n")

    while True:
        try:
            query = input("\n📝 Enter your question (or 'exit'): ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                print("✓ Goodbye!")
                break
            
            if query.lower() == "help":
                print("\nRAG Chatbot Help:")
                print("- Ask questions about RAG and related topics")
                print("- The system retrieves relevant documents and uses Gemini to answer")
                print("- Responses are based on indexed PDF content")
                continue
            
            if not query:
                print("⚠ Please enter a question")
                continue

            # Get search type preference
            search_type = input("Search type (keyword/semantic/hybrid) [default: hybrid]: ").strip().lower()
            if search_type not in ["keyword", "semantic", "hybrid"]:
                search_type = "hybrid"

            # Get top_k preference
            try:
                top_k = int(input("Number of documents to retrieve [default: 5]: ") or "5")
            except ValueError:
                top_k = 5

            # Generate response
            print("\n🔄 Processing...\n")
            response = generate_rag_response(query, search_type=search_type, top_k=top_k, stream=False)
            
            print("\n📚 RESPONSE:")
            print("-" * 80)
            print(response)
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n✓ Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"✗ Error: {str(e)}")


if __name__ == "__main__":
    # Example query
    query = "What is Retrieval-Augmented Generation and how does it work?"
    
    print("\n" + "="*80)
    print("EXAMPLE: RAG Response (Non-Streaming)")
    print("="*80)
    
    # Non-streaming response - returns string directly
    print("\n📚 GENERATING RESPONSE...\n")
    
    response = generate_rag_response(query, search_type="hybrid", top_k=3, stream=False)
    
    print("FINAL RESPONSE:")
    print("="*80)
    print(response)
    print("="*80)
    
    # To test streaming, uncomment below:
    # print("\n" + "="*80)
    # print("EXAMPLE: RAG Response (Streaming)")
    # print("="*80)
    # print("\n🔄 STREAMING RESPONSE:\n")
    # for chunk in generate_rag_response(query, search_type="hybrid", top_k=3, stream=True):
    #     print(chunk, end="", flush=True)
    # print("\n" + "="*80)
    
    # To run interactive mode:
    # interactive_rag()
