# test780 : NoME

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps

import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("No Metadata")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.original_image = None
        self.frequency_image = None
        self.processed_image = None
        self.current_image_path = None
        
        # Noise parameters
        self.gaussian_noise_std = tk.DoubleVar(value=10.0)
        self.frequency_noise_strength = tk.DoubleVar(value=0.1)
        self.affine_angle = tk.DoubleVar(value=2.0)
        self.affine_scale = tk.DoubleVar(value=1.02)
        self.reset_parameters()
        
        self.setup_ui()

    def setup_ui(self):
        # Main frame and configure
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Control panel (left side)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)

        # Image control buttons
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Process Image", command=self.process_image, style="Accent.TButton").pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Save Image", command=self.save_image, style="Accent.TButton").pack(fill=tk.X, pady=5)
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Noise adjustment panel
        noise_frame = ttk.LabelFrame(control_frame, text="Noise Adjustment", padding="10")
        noise_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(noise_frame, text="Gaussian Noise (σ):").pack(anchor=tk.W)
        gaussian_scale = ttk.Scale(noise_frame, from_=0, to=50, variable=self.gaussian_noise_std, orient=tk.HORIZONTAL)
        gaussian_scale.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, textvariable=self.gaussian_noise_std).pack(anchor=tk.W)

        ttk.Label(noise_frame, text="Frequency Noise:").pack(anchor=tk.W, pady=(10, 0))
        freq_scale = ttk.Scale(noise_frame, from_=0, to=1, variable=self.frequency_noise_strength, orient=tk.HORIZONTAL)
        freq_scale.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, textvariable=self.frequency_noise_strength).pack(anchor=tk.W)

        ttk.Label(noise_frame, text="Affine Rotation (°):").pack(anchor=tk.W, pady=(10, 0))
        angle_scale = ttk.Scale(noise_frame, from_=-30, to=30, variable=self.affine_angle, orient=tk.HORIZONTAL)
        angle_scale.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, textvariable=self.affine_angle).pack(anchor=tk.W)

        ttk.Label(noise_frame, text="Affine Scale:").pack(anchor=tk.W, pady=(10, 0))
        scale_scale = ttk.Scale(noise_frame, from_=0.8, to=1.5, variable=self.affine_scale, orient=tk.HORIZONTAL)
        scale_scale.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, textvariable=self.affine_scale).pack(anchor=tk.W)

        ttk.Button(noise_frame, text="Reset Parameters", command=self.reset_parameters).pack(fill=tk.X, pady=(10, 0))

        # Right side panel
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.rowconfigure(1, weight=2)  # upper images
        right_frame.rowconfigure(2, weight=3)  # lower processed image
        right_frame.columnconfigure(0, weight=1)

        # Status bar
        status_frame = ttk.Frame(right_frame)
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.status_label = ttk.Label(status_frame, text="Ready to load image...")
        self.status_label.pack(anchor="w")

        # Upper image display area
        images_frame = ttk.Frame(right_frame)
        images_frame.grid(row=1, column=0, sticky="nsew")
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(1, weight=1)

        # Headers
        ttk.Label(images_frame, text="Original Image", font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(images_frame, text="FFT Frequency Domain", font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=1, padx=5, pady=5)

        # Image panels
        self.original_frame = ttk.LabelFrame(images_frame, padding="5")
        self.original_frame.grid(row=1, column=0, sticky="nsew", padx=2)
        self.frequency_frame = ttk.LabelFrame(images_frame, padding="5")
        self.frequency_frame.grid(row=1, column=1, sticky="nsew", padx=2)

        # Image labels
        self.original_label = ttk.Label(self.original_frame, text="Load an image\nto start")
        self.original_label.pack(expand=True, fill=tk.BOTH)
        self.frequency_label = ttk.Label(self.frequency_frame, text="FFT visualization\nwill appear here\nafter processing")
        self.frequency_label.pack(expand=True, fill=tk.BOTH)

        # Lower processed image display
        images_frame2 = ttk.Frame(right_frame)
        images_frame2.grid(row=2, column=0, sticky="nsew")
        images_frame2.columnconfigure(0, weight=1)
        images_frame2.rowconfigure(1, weight=1)

        ttk.Label(images_frame2, text="Processed Result", font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        self.processed_frame = ttk.LabelFrame(images_frame2, padding="5")
        self.processed_frame.grid(row=1, column=0, sticky="nsew", padx=2)
        self.processed_label = ttk.Label(self.processed_frame, text="Processed image\nwill appear here\nafter processing")
        self.processed_label.pack(expand=True, fill=tk.BOTH)
        
    def reset_parameters(self):
        self.gaussian_noise_std.set(10.0)
        self.frequency_noise_strength.set(0.1)
        self.affine_angle.set(2.0)
        self.affine_scale.set(1.02)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("All files","*.*"), ("png files","*.png"), ("jpg files","*.jpg"), ("jpeg files","*.jpeg"), ("webp files","*.webp"), ("bmp files","*.bmp")])
        if file_path:
            self.load_image_from_path(file_path)
            
    def load_image_from_path(self, path):
        try:
            # Load and convert to RGB
            self.original_image = Image.open(path).convert('RGB')
            self.frequency_image = None
            self.processed_image = None
            self.current_image_path = path.replace("\\", "/")
            
            # Display abd clear other panels
            self.display_image(self.original_image, self.original_label)
            self.frequency_label.config(image='', text="FFT visualization\nwill appear here\nafter processing")
            self.processed_label.config(image='', text="Processed image\nwill appear here\nafter processing")
            
            self.status_label.config(text=f"Loaded: {os.path.basename(path)} ({self.original_image.size[0]}x{self.original_image.size[1]})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def display_image(self, pil_image, label_widget):
        # Calculate display size (max 350x300 for each panel)
        display_width, display_height = 350, 300
        img_width, img_height = pil_image.size
        
        # Calculate scaling factor
        scale = min(display_width / img_width, display_height / img_height)
        if scale < 1:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            display_img = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            display_img = pil_image
            
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(display_img)
        label_widget.config(image=photo, text="")
        label_widget.image = photo  # Keep a reference
        
    def rgb_to_array(self, pil_image):
        return np.array(pil_image)
        
    def array_to_pil(self, array):
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
        
    def create_frequency_visualization(self, image_array):
        if len(image_array.shape) == 3:
            # Use grayscale for frequency visualization
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array
            
        # Apply FFT
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        
        # Create magnitude spectrum for visualization
        magnitude = np.abs(f_shift)
        
        # Apply log transformation to enhance visibility
        magnitude_log = np.log(magnitude + 1)
        
        # Normalize to 0-255 range
        magnitude_normalized = ((magnitude_log - magnitude_log.min()) / (magnitude_log.max() - magnitude_log.min()) * 255)
        
        # Convert to RGB for display
        freq_vis_array = np.stack([magnitude_normalized] * 3, axis=-1)
        
        return self.array_to_pil(freq_vis_array)
        
    def apply_affine_transform(self, image_array):
        angle = self.affine_angle.get()
        scale = self.affine_scale.get()
        
        # Get image center
        h, w = image_array.shape[:2]
        center = (w // 2, h // 2)
        
        # Create affine transformation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Apply transformation to each channel
        if len(image_array.shape) == 3:
            result = np.zeros_like(image_array)
            for i in range(image_array.shape[2]):
                result[:, :, i] = cv2.warpAffine(image_array[:, :, i], matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        else:
            result = cv2.warpAffine(image_array, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        return result
        
    def add_frequency_noise(self, image_array):
        if len(image_array.shape) == 3:
            result = np.zeros_like(image_array)
            for i in range(image_array.shape[2]):
                result[:, :, i] = self._add_freq_noise_channel(image_array[:, :, i])
        else:
            result = self._add_freq_noise_channel(image_array)
        return result
        
    def _add_freq_noise_channel(self, channel):
        # FFT
        f_transform = fft2(channel)
        f_shift = fftshift(f_transform)
        
        # Create noise mask focused on mid-range frequencies
        h, w = channel.shape
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        # Distance from center (normalized)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance / max_distance
        
        # Create noise mask with maximum at mid-range (0.2 to 0.6 normalized distance)
        noise_mask = np.exp(-((normalized_distance - 0.4)**2) / (2 * 0.15**2))
        
        # Generate random noise
        noise_strength = self.frequency_noise_strength.get()
        noise = np.random.normal(0, noise_strength, f_shift.shape) + \
                1j * np.random.normal(0, noise_strength, f_shift.shape)
        
        # Apply noise with mask
        f_shift_noisy = f_shift + noise * noise_mask * np.abs(f_shift) * 0.1
        
        # Inverse FFT
        f_ishift = ifftshift(f_shift_noisy)
        result = np.real(ifft2(f_ishift))
        
        return result
        
    def add_gaussian_noise(self, image_array):
        noise_std = self.gaussian_noise_std.get()
        noise = np.random.normal(0, noise_std, image_array.shape)
        return image_array + noise
        
    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try:
            self.status_label.config(text="Processing image...")
            self.root.update()
            
            # Step 1: Convert to RGB (already done during loading)
            image_array = self.rgb_to_array(self.original_image).astype(np.float64)
            
            # Step 2: Apply affine transformation
            self.status_label.config(text="Applying affine transformation...")
            self.root.update()
            image_array = self.apply_affine_transform(image_array)
            
            # Step 3: Create frequency visualization before adding noise
            self.status_label.config(text="Creating frequency domain visualization...")
            self.root.update()
            self.frequency_image = self.create_frequency_visualization(image_array)
            self.display_image(self.frequency_image, self.frequency_label)
            
            # Step 4: FFT and add frequency noise
            self.status_label.config(text="Adding frequency domain noise...")
            self.root.update()
            image_array = self.add_frequency_noise(image_array)
            
            # Step 5: Add Gaussian noise
            self.status_label.config(text="Adding Gaussian noise...")
            self.root.update()
            image_array = self.add_gaussian_noise(image_array)
            
            # Convert back to PIL image
            self.processed_image = self.array_to_pil(image_array)
            
            # Display processed image
            self.display_image(self.processed_image, self.processed_label)
            
            self.status_label.config(text="Processing complete! All three stages are now visible.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.status_label.config(text="Processing failed!")

    def save_image(self):
        try:
            path = self.current_image_path
            pos = path.rfind("/")
            self.processed_image.save(path[:pos+1] + "[edit] " + path[pos+1:])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            self.status_label.config(text="Save failed!")

if __name__ == "__main__":
    # Create and run the GUI
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
