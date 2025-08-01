# utils/preview.py
import pygame
import numpy as np
from PIL import Image
import io
from pyvirtualdisplay import Display

def render_pygame_preview(code: str, size=(400, 300)):
    display = Display(visible=False, size=size)
    display.start()
    try:
        # Inject save logic
        lines = code.strip().splitlines()
        if "pygame.display" in code:
            # Add screenshot before main loop
            inject = f"""
# Auto-preview frame
pygame.image.save(screen, "assets/pygame_preview.png")
            """
            # Try to inject before main loop
            for i, line in enumerate(lines):
                if "while" in line and "True" in line:
                    lines.insert(i, inject)
                    break
            code = "\n".join(lines)
        
        exec(code, {"__name__": "__main__"}, {})
        
        # Try to load saved image
        try:
            img = Image.open("assets/pygame_preview.png")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            display.stop()
            return buf
        except:
            display.stop()
            return None
    except:
        display.stop()
        return None
