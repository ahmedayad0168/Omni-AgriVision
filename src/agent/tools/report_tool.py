from jinja2 import Template
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ReportTool:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./data/outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, data: str, field_id: int = None, db_manager = None) -> str:
        """Generate a comprehensive farm report from agent analysis data."""
        try:
            # Try to parse as JSON if it's a string
            if isinstance(data, str):
                try:
                    report_data = json.loads(data)
                except json.JSONDecodeError:
                    report_data = {"analysis": data}
            else:
                report_data = data

            # If field_id and db_manager are provided, automatically collect data
            if field_id and db_manager and not report_data.get("field_info"):
                from src.data.models import Scan, Detection, DiseaseRecord, PestRecord, Field
                from datetime import datetime, timedelta

                session = db_manager.get_session()
                try:
                    # Get field info
                    field = session.query(Field).filter(Field.id == field_id).first()
                    if field:
                        report_data["field_info"] = {
                            "field_id": field.id,
                            "field_name": field.field_name,
                            "crop_type": field.crop_type,
                            "area": field.area,
                        }

                    # Get latest scan
                    latest_scan = session.query(Scan).filter(
                        Scan.field_id == field_id
                    ).order_by(Scan.scan_date.desc()).first()

                    if latest_scan:
                        # Get detections and calculate statistics
                        detections = session.query(Detection).filter(
                            Detection.scan_id == latest_scan.id
                        ).all()

                        plants = sum(1 for d in detections if d.object_type == "plant")
                        pests = sum(1 for d in detections if d.object_type == "pest")
                        weeds = sum(1 for d in detections if d.object_type == "weed")
                        diseases = sum(1 for d in detections if d.object_type == "disease")

                        healthy_plants = sum(
                            1 for d in detections
                            if d.object_type == "plant" and "healthy" in d.class_name.lower()
                        )

                        health_score = (healthy_plants / plants * 100) if plants > 0 else 0.0

                        report_data["health_metrics"] = {
                            "health_score": round(health_score, 2),
                            "total_plants": plants,
                            "healthy_plants": healthy_plants,
                            "total_pests": pests,
                            "total_weeds": weeds,
                            "total_diseases": diseases,
                            "scan_date": latest_scan.scan_date.isoformat(),
                        }

                        # Get disease records
                        disease_data = []
                        for d in detections:
                            for dr in session.query(DiseaseRecord).filter(
                                DiseaseRecord.detection_id == d.id
                            ).all():
                                disease_data.append({
                                    "name": dr.disease_name,
                                    "severity": dr.severity,
                                    "severity_level": dr.severity_level,
                                })
                        report_data["diseases"] = disease_data

                        # Get pest records
                        pest_data = []
                        for d in detections:
                            for pr in session.query(PestRecord).filter(
                                PestRecord.detection_id == d.id
                            ).all():
                                pest_data.append({
                                    "name": pr.pest_name,
                                    "count": pr.count,
                                    "density": pr.density,
                                })
                        report_data["pests"] = pest_data

                        # Generate recommendations based on health score
                        recommendations = []
                        if health_score < 80:
                            recommendations.append({
                                "priority": "High",
                                "action": "Implement disease treatment and pest control measures",
                                "rationale": f"Health score of {health_score:.1f}% indicates immediate attention needed"
                            })
                        else:
                            recommendations.append({
                                "priority": "Low",
                                "action": "Continue current farming practices and regular monitoring",
                                "rationale": f"Health score of {health_score:.1f}% indicates good condition"
                            })
                        report_data["recommendations"] = recommendations

                        report_data["summary"] = f"Comprehensive analysis for field {field.field_name if field else field_id} completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

                finally:
                    session.close()

            # Generate HTML report
            html_content = self._generate_html_report(report_data)

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"farm_report_{timestamp}.html"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Report saved to {report_path}")
            return html_content

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report from structured data."""
        
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Omni-AgriVision Farm Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }
                .header h1 {
                    margin: 0;
                    font-size: 2.5em;
                }
                .header .date {
                    margin-top: 10px;
                    opacity: 0.9;
                }
                .section {
                    background: white;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .section h2 {
                    color: #667eea;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                    margin-top: 0;
                }
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .metric-card {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #667eea;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }
                .metric-label {
                    color: #666;
                    margin-top: 5px;
                }
                .alert {
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 4px solid;
                }
                .alert-high {
                    background-color: #fee;
                    border-color: #c33;
                }
                .alert-warning {
                    background-color: #fef9e7;
                    border-color: #f1c40f;
                }
                .alert-info {
                    background-color: #e8f4f8;
                    border-color: #3498db;
                }
                .recommendation {
                    background: #e8f5e8;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #2ecc71;
                    margin: 10px 0;
                }
                .recommendation h4 {
                    margin-top: 0;
                    color: #2ecc71;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #667eea;
                    color: white;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .status-badge {
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 0.9em;
                    font-weight: bold;
                }
                .status-healthy {
                    background-color: #2ecc71;
                    color: white;
                }
                .status-warning {
                    background-color: #f1c40f;
                    color: #333;
                }
                .status-critical {
                    background-color: #e74c3c;
                    color: white;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌾 Omni-AgriVision Farm Report</h1>
                <div class="date">Generated: {{ timestamp }}</div>
            </div>
            
            {% if data.get('field_info') %}
            <div class="section">
                <h2>Field Information</h2>
                <table>
                    <tr><th>Field ID</th><td>{{ data.field_info.get('field_id', 'N/A') }}</td></tr>
                    <tr><th>Field Name</th><td>{{ data.field_info.get('field_name', 'N/A') }}</td></tr>
                    <tr><th>Crop Type</th><td>{{ data.field_info.get('crop_type', 'N/A') }}</td></tr>
                    <tr><th>Area</th><td>{{ data.field_info.get('area', 'N/A') }} hectares</td></tr>
                </table>
            </div>
            {% endif %}
            
            {% if data.get('health_metrics') %}
            <div class="section">
                <h2>Health Metrics</h2>
                <div class="metric-grid">
                    {% for key, value in data.health_metrics.items() %}
                    <div class="metric-card">
                        <div class="metric-value">{{ value }}</div>
                        <div class="metric-label">{{ key|replace('_', ' ')|title }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            
            {% if data.get('diseases') %}
            <div class="section">
                <h2>Disease Analysis</h2>
                <table>
                    <tr>
                        <th>Disease</th>
                        <th>Severity</th>
                        <th>Affected Plants</th>
                        <th>Status</th>
                    </tr>
                    {% for disease in data.diseases %}
                    <tr>
                        <td>{{ disease.name }}</td>
                        <td>{{ disease.severity }}%</td>
                        <td>{{ disease.affected_plants }}</td>
                        <td>
                            <span class="status-badge {% if disease.severity > 30 %}status-critical{% elif disease.severity > 15 %}status-warning{% else %}status-healthy{% endif %}">
                                {% if disease.severity > 30 %}Critical{% elif disease.severity > 15 %}Warning{% else %}Monitor{% endif %}
                            </span>
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if data.get('pests') %}
            <div class="section">
                <h2>Pest Analysis</h2>
                <table>
                    <tr>
                        <th>Pest Type</th>
                        <th>Count</th>
                        <th>Density</th>
                        <th>Status</th>
                    </tr>
                    {% for pest in data.pests %}
                    <tr>
                        <td>{{ pest.name }}</td>
                        <td>{{ pest.count }}</td>
                        <td>{{ pest.density }}</td>
                        <td>
                            <span class="status-badge {% if pest.density > 0.1 %}status-critical{% elif pest.density > 0.05 %}status-warning{% else %}status-healthy{% endif %}">
                                {% if pest.density > 0.1 %}High{% elif pest.density > 0.05 %}Moderate{% else %}Low{% endif %}
                            </span>
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if data.get('alerts') %}
            <div class="section">
                <h2>Alerts & Notifications</h2>
                {% for alert in data.alerts %}
                <div class="alert alert-{{ alert.level|lower }}">
                    <strong>{{ alert.title }}</strong>
                    <p>{{ alert.description }}</p>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if data.get('recommendations') %}
            <div class="section">
                <h2>Recommendations</h2>
                {% for rec in data.recommendations %}
                <div class="recommendation">
                    <h4>{{ rec.priority|title }} Priority</h4>
                    <p>{{ rec.action }}</p>
                    <small>{{ rec.rationale }}</small>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if data.get('weather_forecast') %}
            <div class="section">
                <h2>Weather Forecast</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Temperature</th>
                        <th>Humidity</th>
                        <th>Rainfall</th>
                        <th>Conditions</th>
                    </tr>
                    {% for day in data.weather_forecast %}
                    <tr>
                        <td>{{ day.date }}</td>
                        <td>{{ day.temperature }}°C</td>
                        <td>{{ day.humidity }}%</td>
                        <td>{{ day.rainfall }}mm</td>
                        <td>{{ day.conditions }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if data.get('yield_prediction') %}
            <div class="section">
                <h2>Yield Prediction</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ data.yield_prediction.predicted_yield }}</div>
                        <div class="metric-label">Predicted Yield (tons/hectare)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ data.yield_prediction.confidence }}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                </div>
                <p><strong>Range:</strong> {{ data.yield_prediction.lower_bound }} - {{ data.yield_prediction.upper_bound }} tons/hectare</p>
            </div>
            {% endif %}
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <p>{{ data.get('summary', 'No summary available.') }}</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #666;">
                <p>Generated by Omni-AgriVision AI-Powered Crop Intelligence System</p>
                <p><small>Report ID: {{ report_id }}</small></p>
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        return template.render(
            data=data,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_id=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    
    def generate_pdf_report(self, data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate PDF report using weasyprint."""
        try:
            from weasyprint import HTML
            
            html_content = self._generate_html_report(data)
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(self.output_dir / f"farm_report_{timestamp}.pdf")
            
            HTML(string=html_content).write_pdf(output_path)
            logger.info(f"PDF report saved to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("weasyprint not installed, falling back to HTML only")
            return self.generate_report(json.dumps(data))
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return str(e)