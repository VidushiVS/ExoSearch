import json
import sys

def analyze_json_structure(file_path):
    """Analyze the structure of a JSON file and extract key information."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"File: {file_path}")
        print(f"Total records: {len(data)}")
        print()

        if len(data) == 0:
            print("No data found in file.")
            return

        # Find a record with actual data (not just column definitions)
        sample_record = None
        for record in data:
            # Skip records that look like column definitions
            if isinstance(record, dict) and any(key.startswith('#') for key in record.keys()):
                continue
            if 'pl_name' in record and record['pl_name'] and record['pl_name'] not in ['', 'N/A', 'null']:
                sample_record = record
                break

        if not sample_record:
            # Just take the first non-metadata record
            for record in data:
                if isinstance(record, dict) and record.get('') != []:
                    sample_record = record
                    break

        if not sample_record:
            sample_record = data[0] if data else {}

        print("Available fields in the data:")
        for i, (key, value) in enumerate(sample_record.items()):
            print(f"  {key}: {type(value).__name__} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            if i > 20:  # Limit output
                print("  ... (truncated)")
                break
        print()

        # Look for classification-relevant fields
        classification_fields = ['disposition', 'default_flag', 'tfopwg_disp', 'pl_name']
        print("Classification-relevant fields found:")
        for field in classification_fields:
            if field in sample_record:
                print(f"  {field}: {sample_record[field]}")
        print()

        return sample_record

    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

if __name__ == "__main__":
    files_to_analyze = [
        "k2pandc_2025.10.04_07.10.02.json",
        "TOI_2025.10.04_07.06.07.json"
    ]

    for file_path in files_to_analyze:
        print("=" * 60)
        analyze_json_structure(file_path)
        print()