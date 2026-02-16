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