"""
Computer-Use Toolkit for Kestrion (kestrion[os]).
Allows an agent to natively control the operating system (mouse, keyboard, and screen).
Requires 'pyautogui' and 'Pillow'.
"""

import base64
import io
from typing import Any

from kestrion.agent.decorators import tool

try:
    import pyautogui
    from PIL import ImageGrab
except ImportError:
    pyautogui = None  # type: ignore
    ImageGrab = None  # type: ignore


class OSToolkit:
    """
    Exposes computer use primitives for LLMs (mimicking the Computer-Use API).
    Allows agents to act as autonomous desktop assistants.
    """
    
    def __init__(self):
        if pyautogui is None or ImageGrab is None:
            raise ImportError(
                "The 'os' toolkit requires the 'os' extra dependencies. "
                "Install with `pip install kestrion[os]` (pyautogui and Pillow)."
            )

        # Failsafe for PyAutoGUI: moving mouse to any corner will trigger a failsafe exception
        pyautogui.FAILSAFE = True
        # Adds a 0.1 second pause after every PyAutoGUI call
        pyautogui.PAUSE = 0.1

        @tool(name="screenshot")
        def screenshot() -> dict[str, Any]:
            """
            Take a screenshot of the current screen. Returns a base64 encoded PNG image.
            Use this to see what is currently on the screen before moving the mouse or clicking.
            """
            img = ImageGrab.grab()
            
            # Downscale slightly if too large to save LLM tokens while maintaining layout context
            img.thumbnail((1920, 1920))
            
            buffered = io.BytesIO()
            img.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {
                "status": "success",
                "width": img.width,
                "height": img.height,
                "image_b64": img_str,
            }

        @tool(name="mouse_move")
        def mouse_move(x: int, y: int) -> dict[str, Any]:
            """
            Move the mouse to absolute screen coordinates (x, y).
            """
            pyautogui.moveTo(x, y, duration=0.2)
            return {"status": "success", "x": x, "y": y}

        @tool(name="mouse_click")
        def mouse_click(button: str = "left", double: bool = False) -> dict[str, Any]:
            """
            Click the mouse button ('left', 'right', or 'middle'). Set double=True for a double-click.
            """
            if button not in ["left", "right", "middle"]:
                button = "left"
            if double:
                pyautogui.doubleClick(button=button)
            else:
                pyautogui.click(button=button)
            return {"status": "success", "button": button, "double": double}

        @tool(name="type_text")
        def type_text(text: str) -> dict[str, Any]:
            """
            Type the given text using the keyboard exactly as written.
            """
            pyautogui.write(text, interval=0.01)
            return {"status": "success", "text": text}

        @tool(name="press_key")
        def press_key(keys: str) -> dict[str, Any]:
            """
            Press a specific key or key combination. 
            Pass a single key like 'enter', 'tab', or 'escape'.
            For combinations, pass a space-separated string like 'command space' or 'ctrl c'.
            """
            key_list = keys.split(" ")
            pyautogui.hotkey(*key_list)
            return {"status": "success", "keys": key_list}

        self.tools = [screenshot, mouse_move, mouse_click, type_text, press_key]
