import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import os

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processor")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.original_image = None
        self.frequency_image = None
        self.processed_image = None
        self.current_image_path = None
        
        # Noise parameters
        self.gaussian_noise_std = tk.DoubleVar(value=10.0)
        self.frequency_noise_strength = tk.DoubleVar(value=0.1)
        self.affine_angle = tk.DoubleVar(value=5.0)
        self.affine_scale = tk.DoubleVar(value=1.05)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel (left side)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Load image button
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Load sample images
        ttk.Label(control_frame, text="Sample Images:").pack(anchor=tk.W, pady=(10, 5))
        sample_frame = ttk.Frame(control_frame)
        sample_frame.pack(fill=tk.X, pady=5)
        
        samples = [
            ("Portrait", "samples/realistic_portrait.png"),
            ("Document", "samples/document_text.png"),
            ("Nature", "samples/nature_landscape.png"),
            ("Urban", "samples/urban_scene.png"),
            ("Food", "samples/food_scene.png"),
            ("Cat", "samples/cat_portrait.png")
        ]
        
        for i, (name, path) in enumerate(samples):
            row = i // 2
            col = i % 2
            btn = ttk.Button(sample_frame, text=name, 
                           command=lambda p=path: self.load_sample_image(p))
            btn.grid(row=row, column=col, sticky=(tk.W, tk.E), padx=2, pady=2)
            sample_frame.columnconfigure(col, weight=1)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Process button
        ttk.Button(control_frame, text="Process Image", 
                  command=self.process_image, style="Accent.TButton").pack(fill=tk.X, pady=5)
        
        # Noise adjustment panel
        noise_frame = ttk.LabelFrame(control_frame, text="Noise Adjustment", padding="10")
        noise_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Gaussian noise
        ttk.Label(noise_frame, text="Gaussian Noise (σ):").pack(anchor=tk.W)
        gaussian_scale = ttk.Scale(noise_frame, from_=0, to=50, 
                                 variable=self.gaussian_noise_std, orient=tk.HORIZONTAL)
        gaussian_scale.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, textvariable=self.gaussian_noise_std).pack(anchor=tk.W)
        
        # Frequency noise
        ttk.Label(noise_frame, text="Frequency Noise:").pack(anchor=tk.W, pady=(10, 0))
        freq_scale = ttk.Scale(noise_frame, from_=0, to=1, 
                              variable=self.frequency_noise_strength, orient=tk.HORIZONTAL)
        freq_scale.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, textvariable=self.frequency_noise_strength).pack(anchor=tk.W)
        
        # Affine transformation
        ttk.Label(noise_frame, text="Affine Rotation (°):").pack(anchor=tk.W, pady=(10, 0))
        angle_scale = ttk.Scale(noise_frame, from_=-30, to=30, 
                               variable=self.affine_angle, orient=tk.HORIZONTAL)
        angle_scale.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, textvariable=self.affine_angle).pack(anchor=tk.W)
        
        ttk.Label(noise_frame, text="Affine Scale:").pack(anchor=tk.W, pady=(10, 0))
        scale_scale = ttk.Scale(noise_frame, from_=0.8, to=1.5, 
                               variable=self.affine_scale, orient=tk.HORIZONTAL)
        scale_scale.pack(fill=tk.X, pady=2)
        ttk.Label(noise_frame, textvariable=self.affine_scale).pack(anchor=tk.W)
        
        # Reset button
        ttk.Button(noise_frame, text="Reset Parameters", 
                  command=self.reset_parameters).pack(fill=tk.X, pady=(10, 0))
        
        # Image display area with 3 panels
        images_frame = ttk.Frame(main_frame)
        images_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.columnconfigure(2, weight=1)
        images_frame.rowconfigure(1, weight=1)
        
        # Headers for the three image panels
        ttk.Label(images_frame, text="Original Image", font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(images_frame, text="FFT Frequency Domain", font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(images_frame, text="Processed Result", font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=2, padx=5, pady=5)
        
        # Image panels
        self.original_frame = ttk.LabelFrame(images_frame, padding="5")
        self.original_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2)
        
        self.frequency_frame = ttk.LabelFrame(images_frame, padding="5")
        self.frequency_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2)
        
        self.processed_frame = ttk.LabelFrame(images_frame, padding="5")
        self.processed_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2)
        
        # Image labels
        self.original_label = ttk.Label(self.original_frame, text="Load an image\nto start")
        self.original_label.pack(expand=True, fill=tk.BOTH)
        
        self.frequency_label = ttk.Label(self.frequency_frame, text="FFT visualization\nwill appear here\nafter processing")
        self.frequency_label.pack(expand=True, fill=tk.BOTH)
        
        self.processed_label = ttk.Label(self.processed_frame, text="Processed image\nwill appear here\nafter processing")
        self.processed_label.pack(expand=True, fill=tk.BOTH)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready to load image...")
        self.status_label.pack(anchor=tk.W)
        
    def reset_parameters(self):
        """Reset all noise parameters to defaults"""
        self.gaussian_noise_std.set(10.0)
        self.frequency_noise_strength.set(0.1)
        self.affine_angle.set(5.0)
        self.affine_scale.set(1.05)
        
    def load_image(self):
        """Load an image from file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        
        if file_path:
            self.load_image_from_path(file_path)
    
    def load_sample_image(self, path):
        """Load a sample image"""
        if os.path.exists(path):
            self.load_image_from_path(path)
        else:
            messagebox.showerror("Error", f"Sample image not found: {path}\nPlease run create_samples.py first.")
            
    def load_image_from_path(self, path):
        """Load image from given path"""
        try:
            # Load and convert to RGB
            self.original_image = Image.open(path).convert('RGB')
            self.frequency_image = None
            self.processed_image = None
            self.current_image_path = path
            
            # Display the original image
            self.display_image(self.original_image, self.original_label)
            
            # Clear other panels
            self.frequency_label.config(image='', text="FFT visualization\nwill appear here\nafter processing")
            self.processed_label.config(image='', text="Processed image\nwill appear here\nafter processing")
            
            self.status_label.config(text=f"Loaded: {os.path.basename(path)} ({self.original_image.size[0]}x{self.original_image.size[1]})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def display_image(self, pil_image, label_widget):
        """Display PIL image in the specified label widget"""
        if pil_image is None:
            return
            
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
        """Convert PIL image to numpy array"""
        return np.array(pil_image)
        
    def array_to_pil(self, array):
        """Convert numpy array to PIL image"""
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
        
    def create_frequency_visualization(self, image_array):
        """Create a visualization of the FFT frequency domain"""
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
        magnitude_normalized = ((magnitude_log - magnitude_log.min()) / 
                              (magnitude_log.max() - magnitude_log.min()) * 255)
        
        # Convert to RGB for display
        freq_vis_array = np.stack([magnitude_normalized] * 3, axis=-1)
        
        return self.array_to_pil(freq_vis_array)
        
    def apply_affine_transform(self, image_array):
        """Apply affine transformation to image"""
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
                result[:, :, i] = cv2.warpAffine(image_array[:, :, i], matrix, (w, h), 
                                                borderMode=cv2.BORDER_REFLECT)
        else:
            result = cv2.warpAffine(image_array, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        return result
        
    def add_frequency_noise(self, image_array):
        """Add noise in frequency domain with maximum at mid-range frequencies"""
        if len(image_array.shape) == 3:
            result = np.zeros_like(image_array)
            for i in range(image_array.shape[2]):
                result[:, :, i] = self._add_freq_noise_channel(image_array[:, :, i])
        else:
            result = self._add_freq_noise_channel(image_array)
        return result
        
    def _add_freq_noise_channel(self, channel):
        """Add frequency noise to single channel"""
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
        """Add Gaussian noise to image"""
        noise_std = self.gaussian_noise_std.get()
        noise = np.random.normal(0, noise_std, image_array.shape)
        return image_array + noise
        
    def process_image(self):
        """Apply the complete image processing pipeline"""
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

def main():
    # Create sample images if they don't exist
    if not os.path.exists('samples'):
        print("Creating sample images...")
        import create_samples
        create_samples.create_sample_images()
    
    # Create and run the GUI
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
