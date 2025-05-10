#!/usr/bin/env python3
"""
Test script to check connectivity with different LLM models.
This helps identify API key or network connectivity issues.

Usage:
  python scripts/test_models.py --model gpt-4-turbo
  python scripts/test_models.py --model o3-mini
  python scripts/test_models.py --model all
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Try different imports to handle various environment setups
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Install with: pip install openai")

# Try to import dotenv for environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file if present
except ImportError:
    print("python-dotenv not installed. Environment variables must be set manually.")


def print_separator(char="=", length=50):
    """Print a separator line for better readability."""
    print(char * length)


def test_openai_model(model_name: str, api_key: Optional[str] = None) -> Tuple[bool, str, float]:
    """
    Test connectivity to a specific OpenAI model.

    Args:
        model_name: The name of the model to test
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        (success, message, latency)
    """
    if not OPENAI_AVAILABLE:
        return False, "OpenAI library not installed", 0

    # Ensure API key is set
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY environment variable not set", 0

    print(f"Testing model: {model_name}")
    print(
        f"Using API key: {'*' * (len(os.environ.get('OPENAI_API_KEY', '')) - 8)}...{os.environ.get('OPENAI_API_KEY', '')[-4:] if os.environ.get('OPENAI_API_KEY') else 'None'}"
    )

    # Special handling for local server models
    if model_name.startswith("http://") or model_name.startswith("https://"):
        base_url, model = model_name.rsplit("/", 1)
        client = openai.OpenAI(
            base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "dummy_key")
        )
    else:
        client = openai.OpenAI()

    start_time = time.time()

    try:
        # Simple test message
        completion_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please respond with 'Connection test successful!'"},
            ],
        }

        # Use the correct parameters based on the model
        if model_name.startswith("o3-"):
            completion_kwargs["max_completion_tokens"] = 20
            # Note: temperature is not supported for o3 models
        else:
            completion_kwargs["max_tokens"] = 20
            completion_kwargs["temperature"] = 0

        response = client.chat.completions.create(**completion_kwargs)

        latency = time.time() - start_time
        content = response.choices[0].message.content.strip()

        # Add additional information about the response
        model_used = response.model
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return (
            True,
            f"Response: '{content}'\nModel: {model_used}\nUsage: {json.dumps(usage, indent=2)}",
            latency,
        )

    except Exception as e:
        latency = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)

        if "connect" in error_msg.lower():
            return False, f"Connection Error ({error_type}): {error_msg}", latency
        elif "authentica" in error_msg.lower() or "key" in error_msg.lower():
            return False, f"Authentication Error ({error_type}): {error_msg}", latency
        else:
            return False, f"Error ({error_type}): {error_msg}", latency


def test_models(models: List[str], api_key: Optional[str] = None):
    """Run tests for multiple models and display results."""
    results: Dict[str, Dict] = {}

    for model in models:
        print_separator()
        success, message, latency = test_openai_model(model, api_key)
        results[model] = {"success": success, "message": message, "latency": latency}

        # Print colorized results (using basic ANSI color codes)
        status = "\033[92mSUCCESS\033[0m" if success else "\033[91mFAILED\033[0m"
        print(f"Status: {status}")
        print(f"Latency: {latency:.2f} seconds")
        print(f"Details: {message}")
        print_separator()
        print()

    # Summary
    print_separator(char="-")
    print("TEST SUMMARY")
    print_separator(char="-")
    for model, result in results.items():
        status = "\033[92mOK\033[0m" if result["success"] else "\033[91mFAIL\033[0m"
        print(f"{model}: {status} ({result['latency']:.2f}s)")

    # If any tests failed, suggest troubleshooting steps
    if not all(r["success"] for r in results.values()):
        print_separator(char="-")
        print("\033[93mTROUBLESHOOTING TIPS:\033[0m")
        print("1. Check your API key is correct and has access to the models you're testing")
        print("2. Verify internet connectivity to api.openai.com")
        print("3. If only some models fail, check if your account has access to those models")
        print(
            "4. For Claude models (o3-), ensure you're using Anthropic's API key via the OpenAI API"
        )
        print("5. Set the OPENAI_API_KEY environment variable or use --api-key argument")
        print("6. Check for any proxies or firewalls blocking connections")


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test connectivity to LLM models")
    parser.add_argument(
        "--model", default="all", help="Model to test, or 'all' for a predefined set"
    )
    parser.add_argument("--api-key", help="API key (overrides environment variable)")
    args = parser.parse_args()

    # Define test models
    all_models = [
        "gpt-3.5-turbo",  # Most basic OpenAI model
        "gpt-4-turbo",  # Advanced OpenAI model
        "o3-mini",  # Claude model via OpenAI API
    ]

    # Determine which models to test
    if args.model.lower() == "all":
        test_models(all_models, args.api_key)
    else:
        test_models([args.model], args.api_key)


if __name__ == "__main__":
    main()
