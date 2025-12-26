# download_dependencies.py
import urllib.request
import os
import bz2

def download_shape_predictor():
    """Download dlib's 68-face landmarks predictor"""
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    filename = "shape_predictor_68_face_landmarks.dat"
    
    if os.path.exists(filename):
        print(f"✅ Shape predictor already exists: {filename}")
        return True
    
    print(f"📥 Downloading shape predictor from: {url}")
    print("⚠️ This may take a few minutes (60MB)...")
    
    try:
        # Download the compressed file
        compressed_file = filename + ".bz2"
        print("Downloading...")
        urllib.request.urlretrieve(url, compressed_file)
        
        print("Extracting...")
        # Extract the file
        with bz2.BZ2File(compressed_file) as fr, open(filename, 'wb') as fw:
            fw.write(fr.read())
        
        # Clean up compressed file
        os.remove(compressed_file)
        
        print(f"✅ Successfully downloaded: {filename}")
        print(f"📁 File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("You can manually download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place in the project root directory")
        return False

if __name__ == "__main__":
    download_shape_predictor()