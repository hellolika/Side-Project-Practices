### 1st version: Push Baseline 
- a tool description
- readme attached to the single page as a new tab

### 2nd version: Summary of Changes
1. **Universal Clipboard Helper (`copyTextToClipboard`):** Implemented a fallback mechanism using `document.execCommand('copy')` inside a temporary DOM element when `navigator.clipboard` is restricted by sandboxed iframes or non-secure HTTP contexts.
2. **Fixed `copyAllFilteredLinks()` & `copyToClipboard()`:** Both actions now invoke the fallback helper, providing an alert confirmation or prompt fallback on all devices and browsers.
3. **Refined Search Matching:** Ensured search queries match exact link IDs, channel handles, titles, and descriptions consistently.