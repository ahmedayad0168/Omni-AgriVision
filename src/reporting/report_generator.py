from jinja2 import Template
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    @staticmethod
    def generate_html_report(data: dict, template_str: str = None) -> str:
        if template_str is None:
            template_str = """
            <html>
            <head><title>Farm Report</title></head>
            <body>
            <h1>Farm Report</h1>
            <pre>{{ data | tojson(indent=2) }}</pre>
            </body>
            </html>
            """
        template = Template(template_str)
        return template.render(data=data)

    @staticmethod
    def generate_pdf_report(data: dict, output_path: str):
        # Requires weasyprint
        from weasyprint import HTML
        html = ReportGenerator.generate_html_report(data)
        HTML(string=html).write_pdf(output_path)
        logger.info(f"PDF report saved to {output_path}")