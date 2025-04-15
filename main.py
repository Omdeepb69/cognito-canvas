# main.py
# Main application entry point for Cognito Canvas

import kivy
kivy.require('2.1.0') # Ensure compatible Kivy version

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Line, InstructionGroup
from kivy.properties import ObjectProperty, StringProperty, ListProperty
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.utils import platform

# Image processing and analysis imports
import cv2
import numpy as np
from PIL import Image as PILImage
import io
import threading
import os
import sys
import traceback

# Attempt to import the processor module
try:
    # Ensure the project directory is in the path if running directly
    # This might be needed if 'cognito_canvas' is not installed as a package
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    # Assuming processor.py is inside a 'cognito_canvas' subdirectory
    from cognito_canvas.processor import CanvasProcessor
except ImportError as e:
    print(f"Error importing CanvasProcessor: {e}")
    print("Please ensure 'cognito_canvas/processor.py' exists and all its dependencies (opencv, easyocr, sympy, etc.) are installed.")
    # Define a dummy processor for basic UI functionality if import fails
    class CanvasProcessor:
        def __init__(self):
            print("WARNING: Using dummy CanvasProcessor. Processing will not work.")
        def process_math(self, image_np): return "Error: Math processing unavailable."
        def process_flowchart(self, image_np): return "Error: Flowchart processing unavailable."
        def process_notes(self, image_np): return "Error: Note summarization unavailable."
    # Exit if processor cannot be imported and we want strict dependency
    # sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during processor import: {e}")
    traceback.print_exc()
    sys.exit(1)


# Define the Kivy Language string for the UI layout
# This would typically be in a separate 'cognitocanvas.kv' file
KV_STRING = """
#:import Window kivy.core.window.Window

<InteractiveCanvas>:
    canvas:
        Color:
            rgba: 1, 1, 1, 1 # White background
        Rectangle:
            pos: self.pos
            size: self.size

<MainLayout>:
    orientation: 'vertical'
    canvas_widget: canvas_widget
    result_label: result_label

    InteractiveCanvas:
        id: canvas_widget
        size_hint_y: 0.7

    ScrollView:
        size_hint_y: 0.2
        do_scroll_x: False
        Label:
            id: result_label
            text: 'Draw on the canvas and select an action.'
            text_size: self.width * 0.95, None # Enable text wrapping
            size_hint_y: None
            height: self.texture_size[1] # Adjust height to fit text
            padding: dp(10), dp(10)
            halign: 'left'
            valign: 'top'
            markup: True # Allow simple markup if needed

    BoxLayout:
        size_hint_y: 0.1
        height: dp(50) # Fixed height for buttons
        Button:
            text: 'Solve Math'
            on_press: root.trigger_math_solve()
        Button:
            text: 'Parse Flowchart'
            on_press: root.trigger_flowchart_parse()
        Button:
            text: 'Summarize Notes'
            on_press: root.trigger_note_summary()
        Button:
            text: 'Clear'
            on_press: root.canvas_widget.clear_canvas(); root.result_label.text = 'Canvas cleared.'
"""

Builder.load_string(KV_STRING)

# --- Kivy Widgets ---

class InteractiveCanvas(Widget):
    """
    A widget for drawing freeform lines. Handles touch events for drawing
    and provides methods to capture and clear the canvas content.
    """
    lines = ListProperty([]) # Stores all drawn line instructions

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_line = None
        self._touch_history = {} # Store line instructions per touch

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            with self.canvas:
                Color(0, 0, 0, 1) # Black color for drawing
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=2)
            self._touch_history[touch.uid] = touch.ud['line']
            return True # Consume the touch event
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos) and touch.uid in self._touch_history:
            line = self._touch_history[touch.uid]
            line.points += (touch.x, touch.y)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.uid in self._touch_history:
            line = self._touch_history.pop(touch.uid)
            # Store the completed line instruction group if needed for complex undo/redo
            # For now, just keeping it simple. Lines are drawn directly.
            # self.lines.append(line) # Optional: keep track if needed elsewhere
            return True
        return super().on_touch_up(touch)

    def clear_canvas(self):
        """Clears all drawings from the canvas."""
        # Keep the background rectangle, remove everything else
        self.canvas.clear()
        with self.canvas:
            Color(1, 1, 1, 1) # White background
            Rectangle = kivy.graphics.Rectangle # Local import for canvas context
            Rectangle(pos=self.pos, size=self.size)
        self.lines = []
        self._touch_history = {}
        print("Canvas cleared.")

    def capture_canvas_image(self):
        """
        Captures the current canvas content as a NumPy array (BGR format).
        Exports the widget's texture to an in-memory PNG, then reads it with OpenCV.
        Returns None if the canvas is empty or an error occurs.
        """
        if not self.canvas.children:
             print("Canvas is empty or background only.")
             return None # Avoid capturing an empty canvas

        # Ensure canvas size is positive
        if self.width <= 0 or self.height <= 0:
            print(f"Invalid canvas dimensions: {self.width}x{self.height}")
            return None

        try:
            # Use export_to_png to capture the widget's visual representation
            png_data = self.export_to_png(filename=None) # Returns bytes
            if not png_data:
                print("Error: Failed to export canvas to PNG.")
                return None

            # Read the PNG data from memory using PIL
            pil_image = PILImage.open(io.BytesIO(png_data)).convert('RGB')

            # Convert PIL image to NumPy array (OpenCV uses BGR by default)
            image_np = np.array(pil_image)
            image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            print(f"Canvas captured successfully. Image shape: {image_np_bgr.shape}")
            return image_np_bgr

        except Exception as e:
            print(f"Error capturing canvas image: {e}")
            traceback.print_exc()
            return None


class MainLayout(BoxLayout):
    """
    Root layout containing the canvas, result display, and control buttons.
    Handles triggering the processing actions.
    """
    canvas_widget = ObjectProperty(None)
    result_label = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the processor (can be dummy if import failed)
        self.processor = CanvasProcessor()
        self._processing_lock = threading.Lock() # Prevent concurrent processing

    def _run_processor(self, process_func, image_np):
        """
        Worker function to run a processor method in a separate thread.
        Updates the UI upon completion using Clock.schedule_once.
        """
        result = "Error: Processing failed." # Default error message
        try:
            if image_np is not None:
                print(f"Starting processing function: {process_func.__name__}")
                result = process_func(image_np)
                print(f"Processing finished. Result type: {type(result)}")
            else:
                result = "Error: Could not capture canvas image."
        except Exception as e:
            print(f"Error during processing thread ({process_func.__name__}): {e}")
            traceback.print_exc()
            result = f"Error during processing:\n{e}"
        finally:
            # Schedule UI update on the main thread
            Clock.schedule_once(lambda dt: self._update_result(result))
            # Release the lock
            self._processing_lock.release()
            print("Processing lock released.")

    def _update_result(self, result):
        """Updates the result label on the main Kivy thread."""
        if isinstance(result, (list, tuple)):
             self.result_label.text = "\n".join(map(str, result))
        elif isinstance(result, dict):
             self.result_label.text = "\n".join(f"{k}: {v}" for k, v in result.items())
        else:
             self.result_label.text = str(result)
        print(f"Result label updated: {self.result_label.text[:100]}...") # Log truncated result

    def _start_processing(self, process_func_name):
        """
        Captures the canvas and starts the specified processing function
        in a background thread if not already processing.
        """
        if not self._processing_lock.acquire(blocking=False):
            print("Processing already in progress. Please wait.")
            self.result_label.text = "Processing..."
            return

        print(f"Attempting to start processing: {process_func_name}")
        self.result_label.text = f"Processing {process_func_name.split('_')[-1]}..." # e.g., "Processing math..."

        image_np = self.canvas_widget.capture_canvas_image()

        if image_np is None:
            self.result_label.text = "Failed to capture canvas. Draw something first."
            self._processing_lock.release() # Release lock if capture failed
            print("Processing lock released due to capture failure.")
            return

        # Get the actual processor method
        processor_method = getattr(self.processor, process_func_name, None)

        if processor_method and callable(processor_method):
            # Start the processing in a separate thread
            thread = threading.Thread(
                target=self._run_processor,
                args=(processor_method, image_np),
                daemon=True # Allows app to exit even if thread is running
            )
            thread.start()
        else:
            error_msg = f"Error: Processor method '{process_func_name}' not found or not callable."
            print(error_msg)
            self.result_label.text = error_msg
            self._processing_lock.release() # Release lock if method invalid
            print("Processing lock released due to invalid method.")


    def trigger_math_solve(self):
        """Triggers the math solving process."""
        self._start_processing('process_math')

    def trigger_flowchart_parse(self):
        """Triggers the flowchart parsing process."""
        self._start_processing('process_flowchart')

    def trigger_note_summary(self):
        """Triggers the note summarization process."""
        self._start_processing('process_notes')


class CognitoCanvasApp(App):
    """
    The main Kivy application class for Cognito Canvas.
    """
    def build(self):
        """Builds the application's UI."""
        print("Building Cognito Canvas App...")
        # Set window background color (optional, canvas covers most)
        # Window.clearcolor = (0.9, 0.9, 0.9, 1) # Light grey
        return MainLayout()

    def on_start(self):
        """Called after the build() method is finished."""
        print("Cognito Canvas App started.")
        # You could perform initial checks or setup here if needed
        # e.g., check for processor dependencies, download models if necessary.

    def on_stop(self):
        """Called when the application is closed."""
        print("Cognito Canvas App stopping.")
        # Perform any cleanup here if needed

# --- Main Execution ---

if __name__ == '__main__':
    # Set environment variable for EasyOCR if needed (e.g., for model directory)
    # os.environ['EASYOCR_MODEL_DIR'] = '/path/to/your/models'

    # Handle potential high DPI scaling issues on some platforms
    if platform == 'win':
        try:
            from ctypes import windll, c_int
            windll.shcore.SetProcessDpiAwareness(c_int(2)) # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception as e:
            print(f"Could not set DPI awareness: {e}")

    print("Starting Cognito Canvas...")
    CognitoCanvasApp().run()
    print("Cognito Canvas finished.")