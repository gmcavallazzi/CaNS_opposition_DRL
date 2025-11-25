import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def extract_tensorboard_data(log_dir, output_file):
    """
    Extracts scalar data from TensorBoard log files and saves it to a CSV.
    """
    print(f"Searching for event files in {log_dir}...")
    
    # Find all event files recursively
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print("No event files found.")
        return

    print(f"Found {len(event_files)} event files.")
    
    all_data = []

    for event_file in event_files:
        print(f"Processing {event_file}...")
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            tags = ea.Tags()['scalars']
            
            for tag in tags:
                events = ea.Scalars(tag)
                for event in events:
                    all_data.append({
                        'wall_time': event.wall_time,
                        'step': event.step,
                        'tag': tag,
                        'value': event.value,
                        'file': event_file
                    })
        except Exception as e:
            print(f"Error processing {event_file}: {e}")

    if not all_data:
        print("No scalar data found.")
        return

    print("Converting to DataFrame...")
    df = pd.DataFrame(all_data)
    
    # Pivot table to have tags as columns
    # We group by step and file to handle multiple runs if present, 
    # but usually we want to aggregate or just see the raw data.
    # For simplicity, let's just save the long format which is flexible, 
    # or a wide format if steps are unique.
    
    # Let's try to make a wide format for easier reading: Step vs Tags
    # We might have duplicates if multiple files have the same step (e.g. multiple runs).
    # We'll just save the raw long format and a pivoted version if possible.
    
    # Let's try to make a wide format for easier reading: Step vs Tags
    # We might have duplicates if multiple files have the same step (e.g. multiple runs).
    # We'll just save the raw long format and a pivoted version if possible.
    
    print(f"Saving raw data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Attempt to create a wide format (Step as index, Tags as columns)
    # This assumes we are looking at a single run or don't mind mixing.
    # If there are multiple files, this might get messy, so we'll save a separate file for wide format
    # only if we can distinguish runs.
    
    # Simplified wide format: Group by Step and Tag, taking the mean (in case of duplicates)
    try:
        df_wide = df.pivot_table(index='step', columns='tag', values='value', aggfunc='mean')
        wide_output = output_file.replace('.csv', '_wide.csv')
        print(f"Saving wide format to {wide_output}...")
        df_wide.to_csv(wide_output)
    except Exception as e:
        print(f"Could not create wide format: {e}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract TensorBoard scalars to CSV")
    parser.add_argument("log_dir", type=str, help="Path to the TensorBoard log directory")
    parser.add_argument("--output", type=str, default="tb_metrics.csv", help="Output CSV file name")
    
    args = parser.parse_args()
    
    extract_tensorboard_data(args.log_dir, args.output)
