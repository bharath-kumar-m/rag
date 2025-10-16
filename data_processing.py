import csv
import os

def process_bundles_csv(input_file='bundles.csv', output_file='processed_data.txt'):
    """
    Reads bundles.csv and extracts enhancements, oslist, version, and description
    Organizes data by version with subsections
    """
    
    # Dictionary to store data organized by version
    version_data = {}
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                version = row.get('release_version', 'Unknown Version')
                enhancements = row.get('enhancements', '')
                oslist = row.get('supported_os_list', '')
                description = row.get('description', '')
                
                if version not in version_data:
                    version_data[version] = []
                
                version_data[version].append({
                    'enhancements': enhancements,
                    'supported_os_list': oslist,
                    'description': description
                })
    
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Write organized data to output file
    try:
        with open(output_file, 'a', encoding='utf-8') as outfile:
            for version, items in version_data.items():
                outfile.write(f"\n=== VERSION: {version} ===\n")
                outfile.write("-" * 50 + "\n")
                
                for i, item in enumerate(items, 1):
                    outfile.write(f"\nEntry {i}:\n")
                    outfile.write(f"Enhancements: {item['enhancements']}\n")
                    outfile.write(f"Supported OS List: {item['supported_os_list']}\n")
                    outfile.write(f"Description: {item['description']}\n")
                    outfile.write("\n" + "-" * 30 + "\n")
                
                outfile.write("\n")
        
        print(f"Data successfully processed and appended to {output_file}")
        
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    process_bundles_csv()