import os
import markdown
import subprocess
from datetime import datetime

# Paths - absolute paths for reliability
BASE_DIR = r"c:\Users\HP\Downloads\College\Projects\churn_prediction_project\churn_project"
INPUT_MD = os.path.join(BASE_DIR, "report", "Full_Technical_Manual.md")
TEMP_HTML = os.path.join(BASE_DIR, "report", "temp_report.html")
OUTPUT_PDF = os.path.join(BASE_DIR, "report", "Churn_Analysis_Project_Manual.pdf")

# Professional CSS
CSS = """
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #24292e;
        max-width: 900px;
        margin: 0 auto;
        padding: 45px;
        background-color: white;
    }
    h1, h2, h3 {
        margin-top: 24px;
        margin-bottom: 16px;
        font-weight: 600;
        line-height: 1.25;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 0.3em;
    }
    p { margin-top: 0; margin-bottom: 16px; }
    code {
        padding: 0.2em 0.4em;
        margin: 0;
        font-size: 85%;
        background-color: rgba(27,31,35,0.05);
        border-radius: 3px;
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    }
    pre {
        padding: 16px;
        overflow: auto;
        font-size: 85%;
        line-height: 1.45;
        background-color: #f6f8fa;
        border-radius: 3px;
    }
    table {
        border-spacing: 0;
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 16px;
    }
    table th, table td {
        padding: 6px 13px;
        border: 1px solid #dfe2e5;
    }
    table tr:nth-child(2n) { background-color: #f6f8fa; }
    .footer { font-size: 12px; color: #6a737d; margin-top: 50px; border-top: 1px solid #eaecef; padding-top: 10px; }
</style>
"""

def generate():
    if not os.path.exists(INPUT_MD):
        print(f"Error: Could not find {INPUT_MD}")
        return

    print(f"Reading {INPUT_MD}...")
    with open(INPUT_MD, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Converting Markdown to HTML...")
    html_content = markdown.markdown(text, extensions=['fenced_code', 'tables'])
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Churn Analysis Project Manual</title>
        {CSS}
    </head>
    <body>
        {html_content}
        <div class="footer">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} | Customer Churn Intelligence Platform
        </div>
    </body>
    </html>
    """

    with open(TEMP_HTML, 'w', encoding='utf-8') as f:
        f.write(full_html)

    # Edge command for headless PDF generation
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_path):
        edge_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    print("Running headless browser to generate PDF...")
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={OUTPUT_PDF}",
        TEMP_HTML
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"SUCCESS! PDF generated at: {OUTPUT_PDF}")
    except Exception as e:
        print(f"Error during PDF generation: {e}")
    finally:
        if os.path.exists(TEMP_HTML):
            os.remove(TEMP_HTML)

if __name__ == "__main__":
    generate()
