import os
import re
import csv

def extract_case_ids(base_dir, output_file):
    """
    Extracts TCGA case IDs from files in tcga1, tcga2, and tcga3 directories.
    """
    target_dirs = ['tcga1', 'tcga2', 'tcga3']
    case_ids = []
    
    # Regex pattern to capture the Case ID
    # Pattern looks for "patient_filename: TCGA-XX-XXXX"
    pattern = re.compile(r'patient_filename:\s*(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')

    print(f"Starting extraction from {base_dir}...")

    for dirname in target_dirs:
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue
            
        print(f"Processing directory: {dirname}")
        try:
            files = os.listdir(dir_path)
            for filename in files:
                file_path = os.path.join(dir_path, filename)
                
                # Skip directories if any
                if not os.path.isfile(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        match = pattern.search(content)
                        if match:
                            case_ids.append(match.group(1))
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
        except Exception as e:
             print(f"Error accessing directory {dirname}: {e}")

    # Write to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Case ID'])
            for case_id in case_ids:
                writer.writerow([case_id])
        print(f"Successfully extracted {len(case_ids)} case IDs to {output_file}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

if __name__ == "__main__":
    # Base directory is the directory where the script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_CSV = os.path.join(BASE_DIR, 'tcga_case_ids.csv')
    
    extract_case_ids(BASE_DIR, OUTPUT_CSV)
