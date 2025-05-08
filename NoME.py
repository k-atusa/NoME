from PIL import Image
import os
import sys
import time

def remove_metadata(image_path):
    try:
        dirname, filename = os.path.split(image_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in ['.jpg', '.jpeg', '.png']:
            print(f"[-] Unsupported file format: {ext}")
            return

        timestamp = int(time.time())

        with Image.open(image_path) as img:
            data = list(img.getdata())
            clean_img = Image.new(img.mode, img.size)
            clean_img.putdata(data)

            new_filename = f"{name}-NoME-{timestamp}{ext}"
            new_path = os.path.join(dirname, new_filename)

            if ext in ['.jpg', '.jpeg']:
                clean_img.save(new_path, format='JPEG', quality=95)
            elif ext == '.png':
                clean_img.save(new_path, format='PNG')

            print(f"[+] Metadata removal complete: {new_path}")
    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[*] Usage: NoME.py <image file location>")
    else:
        for image_path in sys.argv[1:]:
            remove_metadata(image_path)
