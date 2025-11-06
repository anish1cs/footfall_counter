# test_api.py
import requests
import time
import os

def test_api():
    # API endpoint
    BASE_URL = 'http://localhost:5000/api'
    
    # 1. Check API health
    response = requests.get(f'{BASE_URL}/health')
    print('Health check:', response.json())
    
    # 2. Upload video for processing
    video_path = 'input/test_video.mp4'  # Make sure this video exists
    if not os.path.exists(video_path):
        print(f"Error: Test video not found at {video_path}")
        return
        
    with open(video_path, 'rb') as video:
        files = {'video': video}
        response = requests.post(f'{BASE_URL}/process-video', files=files)
        
    if response.status_code != 202:
        print('Error uploading video:', response.json())
        return
        
    print('Upload response:', response.json())
    job_id = response.json()['job_id']
    
    # 3. Poll for results
    while True:
        response = requests.get(f'{BASE_URL}/status/{job_id}')
        status = response.json()
        print('Current status:', status)
        
        if status['status'] == 'completed':
            print('\nProcessing completed!')
            print(f"People counted entering: {status['count_in']}")
            print(f"People counted exiting: {status['count_out']}")
            print(f"Net occupancy: {status['net_occupancy']}")
            
            # 4. Download processed video
            response = requests.get(f'{BASE_URL}/video/{job_id}')
            if response.status_code == 200:
                output_path = f"output/processed_{job_id}.avi"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"\nProcessed video saved to: {output_path}")
            break
            
        elif status['status'] == 'failed':
            print('Processing failed:', status['error'])
            break
            
        time.sleep(2)  # Poll every 2 seconds

if __name__ == '__main__':
    test_api()