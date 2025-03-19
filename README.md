# 🛒 Ecommerce Web Scraper using Playwright

![Playwright](https://img.shields.io/badge/Playwright-1.42.0-blue?logo=playwright)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?logo=python)
![License](https://img.shields.io/badge/License-MIT-red)

A **Playwright-based web scraper** that extracts product details (title, price, image URL, and description) from ecommerce websites. Ideal for price tracking, inventory analysis, or building product catalogs.

---

## 🌟 Features

- 🍪 **Cookie Authentication**: Uses browser cookies for seamless session management.
- 📦 **Multi-Format Export**: Saves data to `JSON` (product URLs) and `CSV` (product details).
- 🛡️ **Error Handling**: Gracefully skips missing elements and logs errors.
- 📂 **Structured Output**: Auto-generates organized `data/` folder for results.
- 🚀 **Fast & Reliable**: Built with **Playwright** for high-performance scraping.

---

## 🛠️ Built with Playwright

[Playwright](https://playwright.dev/) is a powerful browser automation library that provides:
- **Cross-browser support**: Works with Chromium, Firefox, and WebKit.
- **Headless and headed modes**: Run browsers in the background or with a visible UI.
- **Automatic waiting**: Waits for elements to load before interacting with them.
- **Network interception**: Mock API responses or block unnecessary resources.
- **Multi-language support**: Available for Python, JavaScript, TypeScript, and more.

---

## 📦 Prerequisites

1. **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
2. **Playwright**: Install browsers using `playwright install`.
3. **Cookies Editor Extension**: [Chrome](https://chrome.google.com/webstore/detail/cookies-editor/) | [Firefox](https://addons.mozilla.org/en-US/firefox/addon/cookies-editor/)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ecommerce-scraper.git
cd ecommerce-scraper
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Playwright Browsers
```bash
playwright install
```

### 4. Set Up Cookies (Optional)
Follow the **[Cookie Setup Guide](#-cookie-setup-guide)** below to generate your `cookies.txt` file.

### 5. Run the Scraper
```bash
python script/scraper.py
```

---

## 🍪 Cookie Setup Guide

### Step 1: Install Cookies Editor Extension
1. Open Chrome or Firefox.
2. Install the **[Cookies Editor](https://chrome.google.com/webstore/detail/cookies-editor/)** extension.

### Step 2: Export Active Cookies
1. Navigate to the target ecommerce website.
2. Log in (if required) and browse to the desired category.
3. Click the **Cookies Editor** extension icon.
4. Click **Export** → **Copy to Clipboard**.

### Step 3: Save Cookies File
1. Create a `config/` folder in your project.
2. Create `cookies.txt` in the `config/` folder.
3. Paste the copied cookies (JSON format) and save.

---

## 📂 Folder Structure

```
ecommerce-scraper/
├── script/
│   └── scraper.py           # Main scraping script
├── config/
│   └── cookies.txt          # Browser cookies (optional)
├── data/                    # Auto-generated during scraping
│   ├── product_urls.json    # All product page URLs
│   └── product_details.csv  # Full product dataset
├── README.md                # You are here!
├── requirements.txt         # Python dependencies
└── .gitignore               # Files/folders to ignore in Git
```

---

## ⚙️ Customization

### Change Target Website
Modify the `homepage_url` and `product_page_url` in `scraper.py`:
```python
homepage_url = "https://www.example.com/"
product_page_url = "https://www.example.com/shop/category/"
```

### Adjust Wait Times
Modify the sleep durations for slower connections:
```python
await page.wait_for_timeout(5000)  # 5-second delay
```

### Add New Fields
Update the `product_details` dictionary in `scraper.py` to include additional fields.

---

## 🚨 Troubleshooting

| Issue                        | Solution                                  |
|------------------------------|-------------------------------------------|
| "Browser binaries missing"   | Run `playwright install`                  |
| "Cookies.txt not found"      | Ensure file is in `config/` folder        |
| Stale Element Errors         | Increase wait times in `scraper.py`       |
| Empty CSV/JSON Files         | Check website structure hasn't changed    |

---

## 📜 Ethical Scraping

This project follows best practices:
- 🐢 Respects `robots.txt` rules.
- ⏳ Includes delays between requests to avoid overloading servers.
- 📉 Limited to 1 concurrent request.
- 🔒 Never stores personal/sensitive data.

---

## 📄 License
MIT License - Use freely but attribute if redistributed.  
**Note**: The scraped data is © the respective ecommerce website. Use responsibly.

---

## 🙏 Credits
- **[Playwright](https://playwright.dev/)** for providing an excellent browser automation library.
- **[Cookies Editor](https://chrome.google.com/webstore/detail/cookies-editor/)** for simplifying cookie management.
