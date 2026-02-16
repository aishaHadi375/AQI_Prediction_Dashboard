# """
# Alert System for Hazardous AQI Levels
# Sends notifications when AQI exceeds dangerous thresholds

# Features:
# - Email alerts
# - SMS alerts (via Twilio)
# - Slack notifications
# - Multiple alert thresholds
# - Alert history logging
# """

# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import json

# # Optional imports
# try:
#     from twilio.rest import Client
#     TWILIO_AVAILABLE = True
# except ImportError:
#     TWILIO_AVAILABLE = False
#     print("⚠️ Twilio not installed. SMS alerts disabled.")

# try:
#     import requests
#     SLACK_AVAILABLE = True
# except ImportError:
#     SLACK_AVAILABLE = False
#     print("⚠️ Requests not installed. Slack alerts disabled.")


# class AQIAlertSystem:
#     """
#     Comprehensive alert system for AQI monitoring
#     """
    
#     # AQI thresholds
#     THRESHOLDS = {
#         'unhealthy': 150,
#         'very_unhealthy': 200,
#         'hazardous': 300
#     }
    
#     def __init__(self, config_file=None):
#         """
#         Initialize alert system with configuration
        
#         Args:
#             config_file: Path to JSON config file with credentials
#         """
#         self.config = self._load_config(config_file)
#         self.alert_history = []
        
#     def _load_config(self, config_file):
#         """Load configuration from file or environment variables"""
#         config = {}
        
#         # Try to load from file
#         if config_file and os.path.exists(config_file):
#             with open(config_file, 'r') as f:
#                 config = json.load(f)
        
#         # Override with environment variables
#         config['email'] = {
#             'smtp_server': os.getenv('SMTP_SERVER', config.get('email', {}).get('smtp_server', 'smtp.gmail.com')),
#             'smtp_port': int(os.getenv('SMTP_PORT', config.get('email', {}).get('smtp_port', 587))),
#             'sender_email': os.getenv('SENDER_EMAIL', config.get('email', {}).get('sender_email', '')),
#             'sender_password': os.getenv('SENDER_PASSWORD', config.get('email', {}).get('sender_password', '')),
#             'recipient_emails': os.getenv('RECIPIENT_EMAILS', config.get('email', {}).get('recipient_emails', [])).split(',') if isinstance(os.getenv('RECIPIENT_EMAILS'), str) else config.get('email', {}).get('recipient_emails', [])
#         }
        
#         config['twilio'] = {
#             'account_sid': os.getenv('TWILIO_ACCOUNT_SID', config.get('twilio', {}).get('account_sid', '')),
#             'auth_token': os.getenv('TWILIO_AUTH_TOKEN', config.get('twilio', {}).get('auth_token', '')),
#             'from_number': os.getenv('TWILIO_FROM_NUMBER', config.get('twilio', {}).get('from_number', '')),
#             'to_numbers': os.getenv('TWILIO_TO_NUMBERS', config.get('twilio', {}).get('to_numbers', [])).split(',') if isinstance(os.getenv('TWILIO_TO_NUMBERS'), str) else config.get('twilio', {}).get('to_numbers', [])
#         }
        
#         config['slack'] = {
#             'webhook_url': os.getenv('SLACK_WEBHOOK_URL', config.get('slack', {}).get('webhook_url', ''))
#         }
        
#         return config
    
#     def check_aqi_threshold(self, current_aqi):
#         """
#         Check if AQI exceeds any threshold
        
#         Returns:
#             tuple: (alert_level, should_alert)
#         """
#         if current_aqi >= self.THRESHOLDS['hazardous']:
#             return 'hazardous', True
#         elif current_aqi >= self.THRESHOLDS['very_unhealthy']:
#             return 'very_unhealthy', True
#         elif current_aqi >= self.THRESHOLDS['unhealthy']:
#             return 'unhealthy', True
#         else:
#             return 'normal', False
    
#     def send_email_alert(self, aqi_value, alert_level, location="Karachi"):
#         """
#         Send email alert
        
#         Args:
#             aqi_value: Current AQI value
#             alert_level: Alert severity level
#             location: Location name
#         """
#         try:
#             email_config = self.config.get('email', {})
            
#             if not email_config.get('sender_email') or not email_config.get('recipient_emails'):
#                 print("⚠️ Email configuration incomplete. Skipping email alert.")
#                 return False
            
#             # Create message
#             msg = MIMEMultipart()
#             msg['From'] = email_config['sender_email']
#             msg['To'] = ', '.join(email_config['recipient_emails'])
#             msg['Subject'] = f"🚨 AQI ALERT - {alert_level.upper()} - {location}"
            
#             # Email body
#             body = f"""
#             <html>
#             <body>
#                 <h2 style="color: {'#dc3545' if alert_level == 'hazardous' else '#fd7e14'};">
#                     Air Quality Alert - {location}
#                 </h2>
                
#                 <p><strong>Alert Level:</strong> {alert_level.replace('_', ' ').upper()}</p>
#                 <p><strong>Current AQI:</strong> <span style="font-size: 24px; color: #dc3545;">{aqi_value:.0f}</span></p>
#                 <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
#                 <h3>Health Advisory:</h3>
#                 <ul>
#                     <li>Everyone should avoid outdoor activities</li>
#                     <li>Keep windows and doors closed</li>
#                     <li>Use air purifiers if available</li>
#                     <li>Wear N95 masks if going outside is necessary</li>
#                 </ul>
                
#                 <p style="color: gray; font-size: 12px;">
#                     This is an automated alert from AQI Monitoring System.
#                 </p>
#             </body>
#             </html>
#             """
            
#             msg.attach(MIMEText(body, 'html'))
            
#             # Send email
#             with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
#                 server.starttls()
#                 server.login(email_config['sender_email'], email_config['sender_password'])
#                 server.send_message(msg)
            
#             print(f"✅ Email alert sent to {len(email_config['recipient_emails'])} recipients")
#             return True
            
#         except Exception as e:
#             print(f"❌ Error sending email: {e}")
#             return False
    
#     def send_sms_alert(self, aqi_value, alert_level, location="Karachi"):
#         """
#         Send SMS alert via Twilio
        
#         Args:
#             aqi_value: Current AQI value
#             alert_level: Alert severity level
#             location: Location name
#         """
#         if not TWILIO_AVAILABLE:
#             print("⚠️ Twilio not available. Skipping SMS alert.")
#             return False
        
#         try:
#             twilio_config = self.config.get('twilio', {})
            
#             if not twilio_config.get('account_sid') or not twilio_config.get('to_numbers'):
#                 print("⚠️ Twilio configuration incomplete. Skipping SMS alert.")
#                 return False
            
#             client = Client(
#                 twilio_config['account_sid'],
#                 twilio_config['auth_token']
#             )
            
#             message_body = (
#                 f"🚨 AQI ALERT - {location}\n"
#                 f"Level: {alert_level.upper()}\n"
#                 f"AQI: {aqi_value:.0f}\n"
#                 f"Avoid outdoor activities!\n"
#                 f"Time: {datetime.now().strftime('%H:%M')}"
#             )
            
#             for to_number in twilio_config['to_numbers']:
#                 message = client.messages.create(
#                     body=message_body,
#                     from_=twilio_config['from_number'],
#                     to=to_number
#                 )
#                 print(f"✅ SMS sent to {to_number}: {message.sid}")
            
#             return True
            
#         except Exception as e:
#             print(f"❌ Error sending SMS: {e}")
#             return False
    
#     def send_slack_alert(self, aqi_value, alert_level, location="Karachi"):
#         """
#         Send Slack notification
        
#         Args:
#             aqi_value: Current AQI value
#             alert_level: Alert severity level
#             location: Location name
#         """
#         if not SLACK_AVAILABLE:
#             print("⚠️ Requests library not available. Skipping Slack alert.")
#             return False
        
#         try:
#             slack_config = self.config.get('slack', {})
#             webhook_url = slack_config.get('webhook_url')
            
#             if not webhook_url:
#                 print("⚠️ Slack webhook URL not configured. Skipping Slack alert.")
#                 return False
            
#             # Color based on severity
#             color_map = {
#                 'unhealthy': '#fd7e14',
#                 'very_unhealthy': '#dc3545',
#                 'hazardous': '#8b0000'
#             }
            
#             payload = {
#                 "text": f"🚨 AQI Alert - {location}",
#                 "attachments": [
#                     {
#                         "color": color_map.get(alert_level, '#dc3545'),
#                         "title": f"Air Quality: {alert_level.replace('_', ' ').upper()}",
#                         "fields": [
#                             {
#                                 "title": "Current AQI",
#                                 "value": f"{aqi_value:.0f}",
#                                 "short": True
#                             },
#                             {
#                                 "title": "Location",
#                                 "value": location,
#                                 "short": True
#                             },
#                             {
#                                 "title": "Time",
#                                 "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                                 "short": False
#                             }
#                         ],
#                         "text": "⚠️ *Health Advisory:* Avoid outdoor activities. Keep windows closed. Use air purifiers."
#                     }
#                 ]
#             }
            
#             response = requests.post(webhook_url, json=payload)
            
#             if response.status_code == 200:
#                 print("✅ Slack alert sent successfully")
#                 return True
#             else:
#                 print(f"❌ Slack alert failed: {response.status_code}")
#                 return False
                
#         except Exception as e:
#             print(f"❌ Error sending Slack notification: {e}")
#             return False
    
#     def send_all_alerts(self, aqi_value, location="Karachi"):
#         """
#         Check AQI and send alerts via all configured channels
        
#         Args:
#             aqi_value: Current AQI value
#             location: Location name
        
#         Returns:
#             dict: Status of each alert channel
#         """
#         alert_level, should_alert = self.check_aqi_threshold(aqi_value)
        
#         if not should_alert:
#             print(f"ℹ️ AQI {aqi_value:.0f} is within safe range. No alerts sent.")
#             return {'alert_triggered': False}
        
#         print(f"\n⚠️ AQI Alert Triggered: {alert_level.upper()} (AQI: {aqi_value:.0f})")
        
#         results = {
#             'alert_triggered': True,
#             'alert_level': alert_level,
#             'aqi_value': aqi_value,
#             'timestamp': datetime.now().isoformat(),
#             'email': self.send_email_alert(aqi_value, alert_level, location),
#             'sms': self.send_sms_alert(aqi_value, alert_level, location),
#             'slack': self.send_slack_alert(aqi_value, alert_level, location)
#         }
        
#         # Log alert
#         self.alert_history.append(results)
        
#         return results
    
#     def save_alert_history(self, filepath="alert_history.json"):
#         """Save alert history to file"""
#         try:
#             with open(filepath, 'w') as f:
#                 json.dump(self.alert_history, f, indent=2)
#             print(f"💾 Alert history saved to {filepath}")
#         except Exception as e:
#             print(f"❌ Error saving alert history: {e}")
    
#     def monitor_predictions(self, predictions_file):
#         """
#         Monitor predictions file and send alerts if needed
        
#         Args:
#             predictions_file: Path to predictions CSV file
#         """
#         try:
#             df = pd.read_csv(predictions_file)
            
#             # Check if any prediction exceeds threshold
#             max_aqi = df['predicted_aqi'].max()
            
#             print(f"\n📊 Monitoring predictions from {predictions_file}")
#             print(f"Max predicted AQI: {max_aqi:.0f}")
            
#             # Send alert if needed
#             results = self.send_all_alerts(max_aqi)
            
#             return results
            
#         except Exception as e:
#             print(f"❌ Error monitoring predictions: {e}")
#             return None


# def main():
#     """
#     Example usage
#     """
#     # Initialize alert system
#     alert_system = AQIAlertSystem()
    
#     # Example 1: Manual alert with specific AQI value
#     print("="*60)
#     print("Testing Manual Alert")
#     print("="*60)
    
#     test_aqi = 220  # Very Unhealthy
#     results = alert_system.send_all_alerts(test_aqi, location="Karachi")
    
#     print("\nAlert Results:")
#     for key, value in results.items():
#         print(f"  {key}: {value}")
    
#     # Example 2: Monitor prediction file
#     print("\n" + "="*60)
#     print("Monitoring Predictions File")
#     print("="*60)
    
#     predictions_path = "../data/predictions/next_3_days_predictions.csv"
#     if os.path.exists(predictions_path):
#         alert_system.monitor_predictions(predictions_path)
#     else:
#         print(f"⚠️ Predictions file not found: {predictions_path}")
    
#     # Save alert history
#     alert_system.save_alert_history()


# if __name__ == "__main__":
#     main()

"""
Discord Alert System for AQI Monitoring
Sends real-time alerts to Discord when AQI exceeds thresholds
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Discord webhook
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests library not available. Install: pip install requests")


class DiscordAQIAlertSystem:
    """
    Discord-based AQI Alert System
    Sends formatted alerts to Discord channels via webhooks
    """
    
    # AQI Thresholds (EPA standards)
    THRESHOLDS = {
        'moderate': 100,
        'unhealthy_sensitive': 150,
        'unhealthy': 200,
        'very_unhealthy': 300,
        'hazardous': 400
    }
    
    def __init__(self, webhook_url=None):
        """
        Initialize Discord Alert System
        
        Args:
            webhook_url: Discord webhook URL (optional, will try .env)
        """
        load_dotenv()
        
        # Get webhook URL from parameter or environment
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        
        if not self.webhook_url:
            print("⚠️ Warning: DISCORD_WEBHOOK_URL not configured")
            print("   Set it in .env file or pass as parameter")
        
        # Alert history
        self.alert_history = []
        
    def get_aqi_category(self, aqi):
        """Determine AQI category and color"""
        if aqi >= self.THRESHOLDS['hazardous']:
            return 'HAZARDOUS', 0x4b0082, '☠️'
        elif aqi >= self.THRESHOLDS['very_unhealthy']:
            return 'VERY UNHEALTHY', 0x8b0000, '😱'
        elif aqi >= self.THRESHOLDS['unhealthy']:
            return 'UNHEALTHY', 0xe74c3c, '😨'
        elif aqi >= self.THRESHOLDS['unhealthy_sensitive']:
            return 'UNHEALTHY FOR SENSITIVE GROUPS', 0xfd7e14, '😷'
        elif aqi >= self.THRESHOLDS['moderate']:
            return 'MODERATE', 0xffc107, '😐'
        else:
            return 'GOOD', 0x28a745, '😊'
    
    def get_health_advisory(self, aqi):
        """Get health advisory message"""
        if aqi >= self.THRESHOLDS['hazardous']:
            return "🆘 **EMERGENCY CONDITIONS!** Stay indoors. Avoid all outdoor exposure. Use air purifiers. Seek medical attention if experiencing symptoms."
        elif aqi >= self.THRESHOLDS['very_unhealthy']:
            return "⛔ **HEALTH ALERT!** Everyone should avoid all outdoor exertion. Move activities indoors. Keep windows closed. Use N95 masks if going outside."
        elif aqi >= self.THRESHOLDS['unhealthy']:
            return "🚨 **UNHEALTHY AIR!** Everyone should limit outdoor activities. Sensitive groups should stay indoors. Reduce prolonged exertion."
        elif aqi >= self.THRESHOLDS['unhealthy_sensitive']:
            return "⚠️ **SENSITIVE GROUPS BEWARE!** Children, elderly, and those with respiratory issues should reduce outdoor activities."
        elif aqi >= self.THRESHOLDS['moderate']:
            return "✅ Air quality acceptable. Unusually sensitive people should consider reducing prolonged outdoor activities."
        else:
            return "🌟 Air quality is excellent! Perfect for all outdoor activities."
    
    def should_send_alert(self, aqi):
        """Determine if alert should be sent based on AQI value"""
        return aqi >= self.THRESHOLDS['unhealthy_sensitive']
    
    def create_embed(self, aqi, location="Karachi", prediction=False):
        """
        Create Discord embed message
        
        Args:
            aqi: AQI value
            location: Location name
            prediction: Whether this is a prediction or current reading
        """
        category, color, emoji = self.get_aqi_category(aqi)
        advisory = self.get_health_advisory(aqi)
        
        embed = {
            "title": f"{emoji} AQI Alert - {location}",
            "description": f"**{'Predicted' if prediction else 'Current'} Air Quality Index**",
            "color": color,
            "fields": [
                {
                    "name": "AQI Value",
                    "value": f"**{aqi:.0f}**",
                    "inline": True
                },
                {
                    "name": "Category",
                    "value": f"**{category}**",
                    "inline": True
                },
                {
                    "name": "Location",
                    "value": location,
                    "inline": True
                },
                {
                    "name": "Health Advisory",
                    "value": advisory,
                    "inline": False
                }
            ],
            "footer": {
                "text": f"AQI Monitoring System • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return embed
    
    def send_discord_message(self, embeds, content=None):
        """
        Send message to Discord
        
        Args:
            embeds: List of embed dictionaries
            content: Optional text content
        """
        if not REQUESTS_AVAILABLE:
            print("❌ requests library not available")
            return False
        
        if not self.webhook_url:
            print("❌ Discord webhook URL not configured")
            return False
        
        try:
            payload = {"embeds": embeds}
            if content:
                payload["content"] = content
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 204:
                print("✅ Discord alert sent successfully")
                return True
            else:
                print(f"❌ Discord webhook failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Discord webhook timeout")
            return False
        except Exception as e:
            print(f"❌ Error sending Discord message: {e}")
            return False
    
    def monitor_current_aqi(self, aqi_value, location="Karachi"):
        """
        Monitor current AQI and send alert if needed
        
        Args:
            aqi_value: Current AQI reading
            location: Location name
        
        Returns:
            dict: Alert status and details
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'aqi_value': aqi_value,
            'location': location,
            'alert_triggered': False,
            'alert_level': None
        }
        
        # Check if alert should be sent
        if self.should_send_alert(aqi_value):
            category, _, _ = self.get_aqi_category(aqi_value)
            result['alert_triggered'] = True
            result['alert_level'] = category
            
            # Create and send alert
            embed = self.create_embed(aqi_value, location, prediction=False)
            success = self.send_discord_message([embed])
            result['sent_successfully'] = success
            
            # Log to history
            self.alert_history.append(result)
            
            print(f"🚨 Alert sent for {location}: AQI {aqi_value:.0f} ({category})")
        else:
            print(f"ℹ️ AQI {aqi_value:.0f} is within safe range. No alert sent.")
        
        return result
    
    def monitor_predictions(self, predictions_file, location="Karachi"):
        """
        Monitor predictions file and send alerts for concerning forecasts
        
        Args:
            predictions_file: Path to predictions CSV file
            location: Location name
        
        Returns:
            dict: Monitoring results
        """
        try:
            if not os.path.exists(predictions_file):
                print(f"❌ Predictions file not found: {predictions_file}")
                return {'error': 'File not found'}
            
            # Load predictions
            df = pd.read_csv(predictions_file)
            
            if 'predicted_aqi' not in df.columns:
                print("❌ 'predicted_aqi' column not found in predictions file")
                return {'error': 'Invalid file format'}
            
            # Find maximum predicted AQI
            max_aqi = df['predicted_aqi'].max()
            max_idx = df['predicted_aqi'].idxmax()
            max_time = df.loc[max_idx, 'datetime'] if 'datetime' in df.columns else 'Unknown'
            
            print(f"\n📊 Prediction Summary:")
            print(f"   Max predicted AQI: {max_aqi:.0f}")
            print(f"   Expected at: {max_time}")
            
            result = {
                'max_predicted_aqi': max_aqi,
                'max_time': str(max_time),
                'location': location,
                'alert_triggered': False
            }
            
            # Send alert if prediction exceeds threshold
            if self.should_send_alert(max_aqi):
                category, _, _ = self.get_aqi_category(max_aqi)
                result['alert_triggered'] = True
                result['alert_level'] = category
                
                # Create prediction alert
                embed = self.create_embed(max_aqi, location, prediction=True)
                
                # Add prediction time field
                embed['fields'].insert(1, {
                    "name": "Expected Time",
                    "value": str(max_time),
                    "inline": True
                })
                
                success = self.send_discord_message(
                    [embed],
                    content=f"⚠️ **FORECAST ALERT** - High AQI expected in {location}"
                )
                result['sent_successfully'] = success
                
                self.alert_history.append(result)
                
                print(f"🚨 Prediction alert sent: AQI {max_aqi:.0f} expected at {max_time}")
            else:
                print(f"✅ All predictions within safe range (max: {max_aqi:.0f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Error monitoring predictions: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def send_simple_message(self, message):
        """
        Send a simple text message to Discord
        
        Args:
            message: Text message to send
        """
        if not REQUESTS_AVAILABLE or not self.webhook_url:
            print("❌ Cannot send message - webhook not configured")
            return False
        
        try:
            payload = {"content": message}
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            return response.status_code == 204
        except Exception as e:
            print(f"❌ Error sending simple message: {e}")
            return False
    
    def send_daily_summary(self, df, location="Karachi"):
        """
        Send a daily AQI summary to Discord
        
        Args:
            df: DataFrame with AQI data (must have 'aqi' and 'datetime' columns)
            location: Location name
        """
        try:
            if 'aqi' not in df.columns:
                print("❌ DataFrame missing 'aqi' column")
                return False
            
            # Get last 24 hours
            last_24h = df.tail(24)
            
            avg_aqi = last_24h['aqi'].mean()
            max_aqi = last_24h['aqi'].max()
            min_aqi = last_24h['aqi'].min()
            current_aqi = last_24h['aqi'].iloc[-1]
            
            category, color, emoji = self.get_aqi_category(avg_aqi)
            
            embed = {
                "title": f"{emoji} Daily AQI Summary - {location}",
                "description": f"**24-Hour Air Quality Report**",
                "color": color,
                "fields": [
                    {
                        "name": "Average AQI",
                        "value": f"**{avg_aqi:.0f}**",
                        "inline": True
                    },
                    {
                        "name": "Current AQI",
                        "value": f"**{current_aqi:.0f}**",
                        "inline": True
                    },
                    {
                        "name": "Category",
                        "value": category,
                        "inline": True
                    },
                    {
                        "name": "Range",
                        "value": f"Min: {min_aqi:.0f} | Max: {max_aqi:.0f}",
                        "inline": False
                    },
                    {
                        "name": "Trend",
                        "value": "📈 Increasing" if current_aqi > avg_aqi else "📉 Decreasing",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": f"Daily Summary • {datetime.now().strftime('%Y-%m-%d')}"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return self.send_discord_message([embed])
            
        except Exception as e:
            print(f"❌ Error sending daily summary: {e}")
            return False
    
    def save_alert_history(self, filepath="alert_history.json"):
        """Save alert history to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
            print(f"💾 Alert history saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving alert history: {e}")
            return False
    
    def load_alert_history(self, filepath="alert_history.json"):
        """Load alert history from JSON file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.alert_history = json.load(f)
                print(f"📂 Loaded {len(self.alert_history)} alerts from history")
                return True
            else:
                print(f"ℹ️ No alert history file found at {filepath}")
                return False
        except Exception as e:
            print(f"❌ Error loading alert history: {e}")
            return False


# Example usage and testing
def main():
    """Test the Discord alert system"""
    print("="*70)
    print("Discord AQI Alert System - Test")
    print("="*70)
    
    # Initialize
    alert_system = DiscordAQIAlertSystem()
    
    # Test with different AQI values
    test_cases = [
        (45, "Good - No alert expected"),
        (120, "Moderate - No alert expected"),
        (165, "Unhealthy for Sensitive - Alert expected"),
        (220, "Unhealthy - Alert expected"),
        (310, "Very Unhealthy - Alert expected")
    ]
    
    print("\n🧪 Testing alert system with various AQI values:\n")
    
    for aqi, description in test_cases:
        print(f"\nTest: {description} (AQI: {aqi})")
        print("-" * 50)
        result = alert_system.monitor_current_aqi(aqi, location="Karachi Test")
        print(f"Result: {'✅ Alert sent' if result['alert_triggered'] else 'ℹ️ No alert'}")
    
    # Save history
    print("\n" + "="*70)
    alert_system.save_alert_history()
    
    print(f"\n📊 Total alerts in session: {len(alert_system.alert_history)}")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            main()
        elif command == "current":
            # Monitor current AQI
            alert_system = DiscordAQIAlertSystem()
            try:
                import hopsworks
                from dotenv import load_dotenv
                load_dotenv()
                
                api_key = os.getenv("HOPSWORKS_API_KEY")
                if api_key:
                    project = hopsworks.login(api_key_value=api_key)
                    fs = project.get_feature_store()
                    fg = fs.get_feature_group("aqi_features", version=2)
                    df = fg.read()
                    
                    if "datetime_str" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime_str"])
                    
                    current_aqi = df.sort_values("datetime")["aqi"].iloc[-1]
                    alert_system.monitor_current_aqi(current_aqi, "Karachi")
                else:
                    print("❌ HOPSWORKS_API_KEY not found")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"Unknown command: {command}")
            print("Usage: python discord_alert_system.py [test|current]")
    else:
        # Default: run tests
        main()