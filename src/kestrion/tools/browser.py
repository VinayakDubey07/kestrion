"""
Playwright-based Browser Automation Agent.
"""

from __future__ import annotations

import base64
from typing import Any

from kestrion.agent.decorators import tool
from kestrion.llm.base import ImageBlock


class BrowserToolkit:
    """
    A stateful browser toolkit for Kestrion agents.
    Provides tools for navigating, extracting text, and interacting with webpages.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._page = None

    async def _get_page(self):
        from playwright.async_api import async_playwright
        
        if self._page is None:
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            if self._browser is None:
                self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._page = await self._browser.new_page()
        return self._page

    async def close(self) -> None:
        """Close the browser resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    def get_tools(self) -> list[Any]:
        """Returns the list of @tool decorated methods to pass to an Agent."""
        return [self.navigate, self.click, self.extract_text, self.screenshot, self.evaluate]

    @tool
    async def navigate(self, url: str) -> str:
        """Navigate the browser to a specific URL."""
        if not url.startswith("http"):
            url = f"https://{url}"
        page = await self._get_page()
        await page.goto(url, wait_until="domcontentloaded")
        return f"Successfully navigated to {page.url}. Title: {await page.title()}"

    @tool
    async def click(self, selector: str) -> str:
        """Click an element on the page using a CSS or XPath selector."""
        page = await self._get_page()
        try:
            await page.locator(selector).first.click(timeout=5000)
            return f"Clicked element matching '{selector}'."
        except Exception as e:
            return f"Failed to click '{selector}': {str(e)}"

    @tool
    async def extract_text(self, selector: str = "body") -> str:
        """Extract the text content from an element matching the selector."""
        page = await self._get_page()
        try:
            text = await page.locator(selector).first.inner_text(timeout=5000)
            return text.strip()[:4000]  # Cap to prevent huge context bloat
        except Exception as e:
            return f"Failed to extract text from '{selector}': {str(e)}"

    @tool
    async def screenshot(self) -> ImageBlock:
        """
        Take a screenshot of the current page.
        Returns a multi-modal ImageBlock that the LLM can see (requires Vision support).
        """
        page = await self._get_page()
        img_bytes = await page.screenshot(type="jpeg", quality=60)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return ImageBlock(data=img_b64, media_type="image/jpeg")

    @tool
    async def evaluate(self, script: str) -> str:
        """Evaluate a javascript expression on the current page."""
        page = await self._get_page()
        try:
            res = await page.evaluate(script)
            return str(res)
        except Exception as e:
            return f"Error evaluating script: {str(e)}"
