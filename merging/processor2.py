
import json as json_lib  # Rename import to avoid conflicts
import os

def create_empty_frame_data():
    """Create empty frame data with zero probabilities and default detection"""
    return {
        "recognition": [0.0] * 100,  # 100 zeros for triplet probabilities
        "detection": [
            {
                "triplet": -1,
                "instrument": [-1, 0, -1, -1, -1, -1]
            }
        ]
    }

def process_and_merge_videos(studio_path, video_configs):
    """
    Process videos, add missing frames, and merge into a single JSON file.
    
    Args:
        studio_path (str): Path to the studio directory
        video_configs (dict): Dictionary with video IDs and their frame ranges
    """
    merged_data = {}
    
    for video_id, max_frame in video_configs.items():
        print(f"\nProcessing video {video_id}")
        video_key = f"VID{video_id}"
        
        # Load the updated model file
        model_file = os.path.join(studio_path, f'updated_model_{video_id}.json')
        try:
            with open(model_file, 'r') as f:
                video_data = json_lib.load(f)
                current_frames = video_data[video_key]
        except FileNotFoundError:
            print(f"Warning: Updated model file not found for video {video_id}")
            current_frames = {}
        except Exception as e:
            print(f"Error loading file {model_file}: {str(e)}")
            current_frames = {}
        
        # Create complete frame range with existing and missing frames
        complete_frames = {}
        for frame_num in range(max_frame + 1):
            frame_id = str(frame_num)
            if frame_id in current_frames:
                complete_frames[frame_id] = current_frames[frame_id]
            else:
                complete_frames[frame_id] = create_empty_frame_data()
        
        # Add to merged data
        merged_data[video_key] = complete_frames
    
    # Save merged data
    output_file = os.path.join(studio_path, 'final.json')
    
    try:
        with open(output_file, 'w') as f:
            json_lib.dump(merged_data, f, indent=2)
        print(f"\nSuccessfully created merged file: {output_file}")
    except Exception as e:
        print(f"Error saving merged file: {str(e)}")

def main():
    # Configuration for each video
    video_configs = {
        '92': 2123,
        '96': 1706,
        '103': 2219,
        '110': 2176,
        '111': 2145
    }

    # Example usage
    studio_path = '/teamspace/studios/this_studio'
    process_and_merge_videos(studio_path, video_configs)

if __name__ == "__main__":
    main()