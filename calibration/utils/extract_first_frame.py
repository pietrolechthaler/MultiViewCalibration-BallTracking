import cv2
import os
import sys

def extract_first_frames(video_dir, output_dir):
    """
    Estrae il primo frame da ogni video MP4 nella directory specificata
    """
    if not os.path.exists(video_dir):
        print(f"Errore: Directory video non trovata: {video_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    
    for video_file in os.listdir(video_dir):
        if video_file.lower().endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            output_file = os.path.splitext(video_file)[0] + '.jpg'
            output_path = os.path.join(output_dir, output_file)
            
            try:
                # Apri il video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Attenzione: Impossibile aprire {video_file}")
                    continue
                
                # Leggi il primo frame
                ret, frame = cap.read()
                if ret:
                    # Salva il frame come JPEG
                    cv2.imwrite(output_path, frame)
                    print(f"Estratto: {output_file}")
                else:
                    print(f"Attenzione: Nessun frame trovato in {video_file}")
                
                # Rilascia le risorse
                cap.release()
                
            except Exception as e:
                print(f"Errore durante l'elaborazione di {video_file}: {str(e)}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Utilizzo: python extract_first_frame.py <video_dir> <output_dir>")
        sys.exit(1)
    
    video_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    extract_first_frames(video_directory, output_directory)