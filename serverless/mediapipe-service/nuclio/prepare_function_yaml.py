#!/usr/bin/env python3
"""
Prepare function.yaml by injecting skeleton spec from JSON file.

This script reads the skeleton configuration from a JSON file and injects it
into function.yaml, using YAML's built-in serialization to handle escaping.
"""

import json
import yaml
import sys
from pathlib import Path

def prepare_function_yaml(spec_file: str, function_yaml: str):
    """Inject spec from JSON file into function.yaml."""

    # Read skeleton spec from JSON file
    spec_path = Path(spec_file)
    if not spec_path.exists():
        print(f"Error: Spec file not found: {spec_file}", file=sys.stderr)
        sys.exit(1)

    with open(spec_path, 'r', encoding='utf-8') as f:
        spec_data = json.load(f)

    # If the JSON contains multiple labels (like raw-editor format),
    # extract "hands-shoulders-skeleton", "person-skeleton", and "hands-skeleton" labels for the function spec
    if isinstance(spec_data, list) and len(spec_data) > 1:
        hands_shoulders_label = [label for label in spec_data if label.get('name') == 'hands-shoulders-skeleton']
        person_label = [label for label in spec_data if label.get('name') == 'person-skeleton']
        hands_label = [label for label in spec_data if label.get('name') == 'hands-skeleton']

        # Include all labels in the spec so users can choose which to use
        all_labels = []
        if hands_shoulders_label:
            all_labels.extend(hands_shoulders_label)
            print(f"  - Extracted 'hands-shoulders-skeleton' label from multi-label config")
        if person_label:
            all_labels.extend(person_label)
            print(f"  - Extracted 'person-skeleton' label from multi-label config")
        if hands_label:
            all_labels.extend(hands_label)
            print(f"  - Extracted 'hands-skeleton' label from multi-label config")

        if all_labels:
            spec_data = all_labels
        elif person_label:
            # Fallback to just person-skeleton if others not found
            spec_data = person_label

    # Convert to JSON string (this is what CVAT expects)
    spec_json = json.dumps(spec_data, ensure_ascii=False)

    # Read and parse YAML file
    yaml_path = Path(function_yaml)
    if not yaml_path.exists():
        print(f"Error: Function YAML not found: {function_yaml}", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Update spec in annotations - YAML will handle escaping automatically
    if 'metadata' not in data or 'annotations' not in data['metadata']:
        print("Error: function.yaml missing metadata.annotations", file=sys.stderr)
        sys.exit(1)

    data['metadata']['annotations']['spec'] = spec_json

    # Use YAML's representer to write as a quoted string
    # We'll use a custom representer to ensure it's written as a single-line quoted string
    class QuotedString(str):
        pass

    def quoted_string_representer(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

    yaml.add_representer(QuotedString, quoted_string_representer)

    # Convert spec_json to QuotedString so it gets quoted style
    data['metadata']['annotations']['spec'] = QuotedString(spec_json)

    # Write back using YAML dumper with default_flow_style=False to preserve structure
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=float('inf'))

    print(f"✓ Updated {function_yaml} with spec from {spec_file}")
    print(f"  - Spec length: {len(spec_json)} characters")
    print(f"  - Contains 'data-label-name': {'data-label-name' in spec_json}")
    print(f"  - Contains unescaped '<': {'<' in spec_json}")


if __name__ == "__main__":
    # Default paths relative to script location
    script_dir = Path(__file__).parent
    # Use raw-editor.json as source (contains hands-shoulders, person, and hands skeletons)
    default_spec = script_dir.parent / "mediapipe-skeletons-raw-editor.json"
    default_yaml = script_dir / "function.yaml"

    if len(sys.argv) > 1:
        spec_file = sys.argv[1]
    else:
        spec_file = str(default_spec)

    if len(sys.argv) > 2:
        function_yaml = sys.argv[2]
    else:
        function_yaml = str(default_yaml)

    prepare_function_yaml(spec_file, function_yaml)
