import os
import cv2
import numpy as np
import threading
import tempfile
from datetime import datetime
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.utils import platform
from kivy.core.window import Window
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, BooleanProperty
from kivy.lang import Builder

try:
    from plyer import filechooser
except ImportError:
    pass

from cognito_canvas.processor import CanvasProcessor, process_image_for_ocr

class ProcessingPopup(Popup):
    def __init__(self, **kwargs):
        super(ProcessingPopup, self).__init__(**kwargs)
        self.title = "Processing..."
        self.size_hint = (0.6, 0.3)
        self.auto_dismiss = False
        
        layout = BoxLayout(orientation='vertical', padding=20)
        self.label = Label(text="Processing your image. Please wait...", font_size=18)
        layout.add_widget(self.label)
        self.content = layout

class ResultViewer(ScrollView):
    text = StringProperty("")
    
    def __init__(self, **kwargs):
        super(ResultViewer, self).__init__(**kwargs)
        self.text_widget = TextInput(text=self.text, readonly=True, multiline=True, 
                                     size_hint=(1, None), font_size=16)
        self.text_widget.bind(minimum_height=self.text_widget.setter('height'))
        self.add_widget(self.text_widget)
    
    def update_text(self, text):
        self.text_widget.text = text

class ImagePreview(Image):
    def __init__(self, **kwargs):
        super(ImagePreview, self).__init__(**kwargs)
        self.allow_stretch = True
        self.keep_ratio = True
        self.size_hint = (1, 1)

class CognitoCanvasApp(App):
    processor = ObjectProperty(None)
    current_image = ObjectProperty(None)
    processed_image = ObjectProperty(None)
    processing_thread = ObjectProperty(None)
    current_mode = StringProperty('notes')
    processing_complete = BooleanProperty(False)
    
    def build(self):
        self.processor = CanvasProcessor(use_gemini=True)
        self.title = 'Cognito Canvas'
        return Builder.load_file('cognito.kv')
    
    def on_start(self):
        self.root.ids.status_label.text = "Ready. Select an image or use the capture button."
        Window.bind(on_resize=self._on_window_resize)
    
    def _on_window_resize(self, instance, width, height):
        if self.root.ids.image_preview.texture:
            self._update_texture(self.processed_image)
    
    def show_file_chooser(self):
        try:
            filechooser.open_file(on_selection=self._handle_selection, 
                                  filters=[("Image Files", "*.png", "*.jpg", "*.jpeg")])
        except:
            # Fallback for desktop
            from tkinter import Tk
            from tkinter.filedialog import askopenfilename
            Tk().withdraw()
            path = askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
            if path:
                self._handle_selection([path])
    
    def _handle_selection(self, selection):
        if selection:
            self.load_image(selection[0])
    
    def load_image(self, path):
        try:
            if not os.path.exists(path):
                self.root.ids.status_label.text = f"Error: File not found: {path}"
                return
            
            self.current_image = cv2.imread(path)
            if self.current_image is None:
                self.root.ids.status_label.text = f"Error: Could not load image: {path}"
                return
            
            self.processed_image = self.current_image.copy()
            self._update_texture(self.processed_image)
            self.root.ids.status_label.text = f"Image loaded: {os.path.basename(path)}"
            self.processing_complete = False
            self.root.ids.result_viewer.update_text("")
        except Exception as e:
            self.root.ids.status_label.text = f"Error loading image: {str(e)}"
    
    def _update_texture(self, image):
        if image is None:
            return
        
        # Resize for display while maintaining aspect ratio
        max_height = int(self.root.ids.image_preview.height)
        scale_factor = max_height / image.shape[0] if max_height > 0 else 1
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        
        if width > 0 and height > 0:
            display_image = cv2.resize(image, (width, height))
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            
            texture = Texture.create(size=(display_image.shape[1], display_image.shape[0]), colorfmt='rgb')
            texture.blit_buffer(display_image.flatten(), colorfmt='rgb', bufferfmt='ubyte')
            
            self.root.ids.image_preview.texture = texture
    
    def enhance_image(self):
        if self.current_image is None:
            self.root.ids.status_label.text = "No image loaded to enhance."
            return
        
        try:
            self.processed_image = process_image_for_ocr(self.current_image, enhance_contrast=True)
            if self.processed_image is not None:
                # Convert back to BGR for display
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
                self._update_texture(self.processed_image)
                self.root.ids.status_label.text = "Image enhanced for OCR."
            else:
                self.root.ids.status_label.text = "Image enhancement failed."
        except Exception as e:
            self.root.ids.status_label.text = f"Error enhancing image: {str(e)}"
    
    def capture_image(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])
            try:
                from plyer import camera
                temp_file = os.path.join(tempfile.gettempdir(), f"cognito_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                camera.take_picture(temp_file, self.on_camera_capture)
            except Exception as e:
                self.root.ids.status_label.text = f"Error capturing image: {str(e)}"
        else:
            self.root.ids.status_label.text = "Camera capture not supported on this platform."
    
    def on_camera_capture(self, file_path):
        if os.path.exists(file_path):
            self.load_image(file_path)
        else:
            self.root.ids.status_label.text = "Camera capture failed."
    
    def set_mode(self, mode):
        self.current_mode = mode
        self.root.ids.status_label.text = f"Mode set to: {mode.capitalize()}"
        # Reset results when changing modes
        if self.processing_complete:
            self.processing_complete = False
            self.root.ids.result_viewer.update_text("")
    
    def process_image(self):
        if self.current_image is None:
            self.root.ids.status_label.text = "Please load an image first."
            return
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.root.ids.status_label.text = "Processing is already in progress."
            return
        
        # Show processing popup
        self.processing_popup = ProcessingPopup()
        self.processing_popup.open()
        
        # Start processing in a separate thread to avoid UI freeze
        self.processing_thread = threading.Thread(target=self._process_in_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_in_thread(self):
        try:
            image_to_process = self.processed_image if self.processed_image is not None else self.current_image
            
            if self.current_mode == 'math':
                result = self.processor.process_math(image_to_process)
            elif self.current_mode == 'flowchart':
                result = self.processor.process_flowchart(image_to_process)
            else:  # Default to notes
                result = self.processor.process_notes(image_to_process)
            
            # Update UI in the main thread
            Clock.schedule_once(lambda dt: self._processing_finished(result), 0)
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            Clock.schedule_once(lambda dt: self._processing_error(error_msg), 0)
    
    def _processing_finished(self, result):
        self.processing_popup.dismiss()
        self.root.ids.result_viewer.update_text(result)
        self.root.ids.status_label.text = f"Processing complete ({self.current_mode} mode)."
        self.processing_complete = True
    
    def _processing_error(self, error_msg):
        self.processing_popup.dismiss()
        self.root.ids.status_label.text = error_msg
        self.root.ids.result_viewer.update_text(f"Processing Error:\n{error_msg}")
    
    def save_results(self):
        if not self.processing_complete:
            self.root.ids.status_label.text = "No results to save."
            return
        
        try:
            results_text = self.root.ids.result_viewer.text_widget.text
            if not results_text:
                self.root.ids.status_label.text = "No results to save."
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cognito_results_{self.current_mode}_{timestamp}.txt"
            
            # Save results
            with open(filename, "w") as f:
                f.write(results_text)
            
            self.root.ids.status_label.text = f"Results saved to {filename}"
        except Exception as e:
            self.root.ids.status_label.text = f"Error saving results: {str(e)}"

if __name__ == '__main__':
    CognitoCanvasApp().run()
