#!/usr/bin/env python3
"""
Helper script to save Python dict as JSON for data parsing.

Usage in your Python script:
    import json

    # Your dict from wherever you got it
    inputs = {...}

    # Save it
    with open("my_data.json", "w") as f:
        json.dump(inputs, f, indent=2)
"""

import json
import argparse


def save_dict_as_json(data_dict, output_path):
    """
    Save Python dictionary as formatted JSON file.

    Args:
        data_dict: Dictionary to save
        output_path: Output JSON file path
    """
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"✓ Saved dict as JSON to: {output_path}")


def main():
    """
    Example usage demonstrating how to save a dict.
    """
    print("="*60)
    print("Dict to JSON Helper")
    print("="*60)
    print()
    print("In your Python code, use:")
    print()
    print("  import json")
    print()
    print("  # If you have: inputs['json']")
    print("  data = inputs['json']")
    print()
    print("  # Save as JSON")
    print("  with open('my_data.json', 'w') as f:")
    print("      json.dump({'json': data}, f, indent=2)")
    print()
    print("Then run:")
    print("  python scripts/parse_nested_json.py \\")
    print("      --input my_data.json \\")
    print("      --output data_config.json")
    print()
    print("="*60)


if __name__ == "__main__":
    main()
