import sys
import os
from app import check_services, create_ui
from generation import validate_gemini_api_key

def main():
    """
    Main entry point for the LocalRAG application.
    Launches the Gradio web interface.
    """
    print("\n" + "="*80)
    print("🚀 Starting LocalRAG Q&A System...")
    print("="*80)
    print("\n✓ Environment Checks:")
    print("  - Verifying GEMINI_API_KEY...")
    print("  - Checking OpenSearch connection...")
    print("  - Initializing Gradio UI...\n")
    
    try:
        validate_gemini_api_key()
        services_ready, service_message = check_services()
        if services_ready:
            print(f"✓ {service_message}")
        else:
            print(f"⚠ {service_message}")
            print("  The UI will still launch, but retrieval needs these services.")

        # Create and launch the Gradio interface
        demo = create_ui()
        print("✓ UI Created Successfully\n")
        print("💡 Launching Gradio Server...")
        print("   URL: http://localhost:7860")
        print("   Share: False\n")
        print("⚠️  FREE TIER LIMITS:")
        print("   - 15 requests per minute (RPM)")
        print("   - 1 million tokens per day")
        print("   - Rate limiting enabled in requests\n")
        print("Press CTRL+C to stop the server.\n")
        print("="*80 + "\n")
        
# Launch with queue for better streaming support
        demo.queue().launch(
            share=False,
            server_name="0.0.0.0",
            server_port=int(os.getenv("APP_PORT", "7860")),
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n✓ Server stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error starting LocalRAG: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check GEMINI_API_KEY is set in .env")
        print("  2. Verify OpenSearch is running: docker compose up -d")
        print("  3. Verify Ollama is running: docker ps | grep ollama")
        sys.exit(1)


if __name__ == "__main__":
    main()
