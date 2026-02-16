import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from joblib import load
import os
import warnings
warnings.filterwarnings('ignore')

# Optional: Hopsworks integration
try:
    import hopsworks
    from dotenv import load_dotenv
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    print("⚠️ Hopsworks not available. Will use local data if available.")


class AQIShapAnalyzer:
    """
    Comprehensive SHAP analysis for AQI prediction models
    """
    
    def __init__(self, model_name="Random_Forest", data_source="hopsworks"):
        """
        Initialize SHAP analyzer
        
        Args:
            model_name: Name of model to analyze (Random_Forest, XGBoost, etc.)
            data_source: 'hopsworks' or 'local'
        """
        self.model_name = model_name
        self.data_source = data_source
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.explainer = None
        self.shap_values = None
        
        print(f"🔍 Initializing SHAP Analyzer for {model_name}...")
        
    def load_model(self):
        """Load trained model from local storage"""
        try:
            model_path = f"../models/{self.model_name.lower()}.pkl"
            if not os.path.exists(model_path):
                model_path = f"models/{self.model_name.lower()}.pkl"
            
            self.model = load(model_path)
            print(f"✅ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_data(self):
        """Load data from Hopsworks or local file"""
        try:
            if self.data_source == "hopsworks" and HOPSWORKS_AVAILABLE:
                return self._load_from_hopsworks()
            else:
                return self._load_from_local()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _load_from_hopsworks(self):
        """Load data from Hopsworks Feature Store"""
        try:
            load_dotenv()
            api_key = os.getenv("HOPSWORKS_API_KEY")
            
            if not api_key:
                print("⚠️ Hopsworks API key not found. Trying local data...")
                return self._load_from_local()
            
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            fg = fs.get_feature_group("aqi_features", version=2)
            df = fg.read()
            
            print(f"✅ Fetched {len(df)} rows from Hopsworks")
            
            # Prepare data
            if "datetime_str" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime_str"])
                df.drop(columns=["datetime_str"], inplace=True)
            
            df = df.sort_values("datetime").reset_index(drop=True)
            
            # Remove leakage features
            leakage_features = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
            df.drop(columns=[c for c in leakage_features if c in df.columns], 
                   inplace=True, errors='ignore')
            
            # Split features and target
            X = df.drop(columns=["aqi", "datetime"], errors="ignore")
            y = df["aqi"]
            
            # Train-test split (80/20)
            split_index = int(len(df) * 0.8)
            self.X_train = X.iloc[:split_index]
            self.X_test = X.iloc[split_index:]
            self.y_test = y.iloc[split_index:]
            self.feature_names = X.columns.tolist()
            
            print(f"✅ Data prepared: Train={len(self.X_train)}, Test={len(self.X_test)}")
            return True
            
        except Exception as e:
            print(f"⚠️ Hopsworks loading failed: {e}")
            return self._load_from_local()
    
    def _load_from_local(self):
        """Load data from local CSV file"""
        try:
            # Try multiple possible paths
            possible_paths = [
                "../data/final/final_selected_features.csv",
                "data/final/final_selected_features.csv",
                "../data/final/clean_merged_karachi.csv",
                "data/final/clean_merged_karachi.csv"
            ]
            
            df = None
            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    print(f"✅ Loaded data from: {path}")
                    break
            
            if df is None:
                print("❌ No local data file found")
                return False
            
            # Prepare data
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            
            df = df.sort_values("datetime").reset_index(drop=True)
            
            # Remove leakage features
            leakage_features = ["aqi_rolling_24h", "aqi_lag_1h", "high_pollution_flag"]
            df.drop(columns=[c for c in leakage_features if c in df.columns], 
                   inplace=True, errors='ignore')
            
            # Split features and target
            X = df.drop(columns=["aqi", "datetime", "datetime_str"], errors="ignore")
            y = df["aqi"]
            
            # Train-test split (80/20)
            split_index = int(len(df) * 0.8)
            self.X_train = X.iloc[:split_index]
            self.X_test = X.iloc[split_index:]
            self.y_test = y.iloc[split_index:]
            self.feature_names = X.columns.tolist()
            
            print(f"✅ Data prepared: Train={len(self.X_train)}, Test={len(self.X_test)}")
            return True
            
        except Exception as e:
            print(f"❌ Local loading failed: {e}")
            return False
    
    def create_explainer(self, background_samples=100):
        """
        Create SHAP explainer based on model type
        
        Args:
            background_samples: Number of samples for background dataset
        """
        try:
            print(f"\n🔧 Creating SHAP explainer for {self.model_name}...")
            
            # Sample background data for faster computation
            if len(self.X_train) > background_samples:
                background = shap.sample(self.X_train, background_samples)
            else:
                background = self.X_train
            
            # Choose appropriate explainer based on model type
            model_type = type(self.model).__name__
            
            if "Forest" in model_type or "XGB" in model_type or "Gradient" in model_type:
                # Tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                print("✅ Using TreeExplainer (fast and exact)")
            else:
                # Other models (Ridge, etc.)
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    background
                )
                print("✅ Using KernelExplainer (model-agnostic)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating explainer: {e}")
            return False
    
    def compute_shap_values(self, n_samples=100):
        """
        Compute SHAP values for test set
        
        Args:
            n_samples: Number of test samples to explain
        """
        try:
            print(f"\n⚙️ Computing SHAP values for {n_samples} samples...")
            
            # Sample test data
            if len(self.X_test) > n_samples:
                X_explain = self.X_test.sample(n_samples, random_state=42)
            else:
                X_explain = self.X_test
            
            # Compute SHAP values
            self.shap_values = self.explainer.shap_values(X_explain)
            self.X_explain = X_explain
            
            print(f"✅ SHAP values computed for {len(X_explain)} samples")
            return True
            
        except Exception as e:
            print(f"❌ Error computing SHAP values: {e}")
            return False
    
    def plot_summary(self, save_path=None):
        """
        Create SHAP summary plot (feature importance)
        """
        try:
            print("\n📊 Creating SHAP summary plot...")
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values, 
                self.X_explain,
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f"SHAP Feature Importance - {self.model_name}", 
                     fontsize=16, pad=20)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating summary plot: {e}")
    
    def plot_bar(self, save_path=None):
        """
        Create SHAP bar plot (mean absolute SHAP values)
        """
        try:
            print("\n📊 Creating SHAP bar plot...")
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values, 
                self.X_explain,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False
            )
            plt.title(f"SHAP Feature Importance (Bar) - {self.model_name}", 
                     fontsize=16, pad=20)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating bar plot: {e}")
    
    def plot_waterfall(self, instance_index=0, save_path=None):
        """
        Create waterfall plot for a single prediction
        
        Args:
            instance_index: Index of instance to explain
        """
        try:
            print(f"\n📊 Creating waterfall plot for instance {instance_index}...")
            
            # Create explanation object
            shap_exp = shap.Explanation(
                values=self.shap_values[instance_index],
                base_values=self.explainer.expected_value,
                data=self.X_explain.iloc[instance_index].values,
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(shap_exp, show=False)
            plt.title(f"SHAP Waterfall Plot - Instance {instance_index}", 
                     fontsize=16, pad=20)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating waterfall plot: {e}")
    
    def plot_force(self, instance_index=0, save_path=None):
        """
        Create force plot for a single prediction
        
        Args:
            instance_index: Index of instance to explain
        """
        try:
            print(f"\n📊 Creating force plot for instance {instance_index}...")
            
            shap.initjs()
            
            force_plot = shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[instance_index],
                self.X_explain.iloc[instance_index],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            plt.title(f"SHAP Force Plot - Instance {instance_index}", 
                     fontsize=16, pad=20)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating force plot: {e}")
    
    def plot_dependence(self, feature_name, interaction_feature=None, save_path=None):
        """
        Create SHAP dependence plot
        
        Args:
            feature_name: Feature to plot
            interaction_feature: Feature to color by (auto-detected if None)
        """
        try:
            print(f"\n📊 Creating dependence plot for {feature_name}...")
            
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(
                feature_name,
                self.shap_values,
                self.X_explain,
                interaction_index=interaction_feature,
                show=False
            )
            plt.title(f"SHAP Dependence Plot - {feature_name}", 
                     fontsize=16, pad=20)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating dependence plot: {e}")
    
    def get_feature_importance(self):
        """
        Extract feature importance as DataFrame
        """
        try:
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"❌ Error extracting feature importance: {e}")
            return None
    
    def generate_report(self, output_dir="shap_analysis"):
        """
        Generate comprehensive SHAP analysis report
        """
        try:
            print(f"\n📋 Generating comprehensive SHAP report...")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Summary plot
            self.plot_summary(
                save_path=os.path.join(output_dir, "01_summary_plot.png")
            )
            
            # 2. Bar plot
            self.plot_bar(
                save_path=os.path.join(output_dir, "02_bar_plot.png")
            )
            
            # 3. Waterfall plots for top 3 predictions
            for i in range(min(3, len(self.X_explain))):
                self.plot_waterfall(
                    instance_index=i,
                    save_path=os.path.join(output_dir, f"03_waterfall_instance_{i}.png")
                )
            
            # 4. Dependence plots for top 5 features
            importance_df = self.get_feature_importance()
            if importance_df is not None:
                top_features = importance_df['feature'].head(5).tolist()
                
                for i, feature in enumerate(top_features):
                    if feature in self.X_explain.columns:
                        self.plot_dependence(
                            feature_name=feature,
                            save_path=os.path.join(output_dir, f"04_dependence_{i+1}_{feature}.png")
                        )
            
            # 5. Save feature importance CSV
            if importance_df is not None:
                csv_path = os.path.join(output_dir, "feature_importance.csv")
                importance_df.to_csv(csv_path, index=False)
                print(f"💾 Feature importance saved to: {csv_path}")
            
            print(f"\n✅ Full report generated in: {output_dir}/")
            print(f"   Files created: summary, bar, waterfall, dependence plots + CSV")
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    def run_full_analysis(self, output_dir="shap_analysis"):
        """
        Run complete SHAP analysis pipeline
        """
        print("="*60)
        print("🚀 Starting Full SHAP Analysis Pipeline")
        print("="*60)
        
        # Load model
        if not self.load_model():
            return False
        
        # Load data
        if not self.load_data():
            return False
        
        # Create explainer
        if not self.create_explainer():
            return False
        
        # Compute SHAP values
        if not self.compute_shap_values():
            return False
        
        # Generate report
        self.generate_report(output_dir)
        
        print("\n" + "="*60)
        print("✅ SHAP Analysis Complete!")
        print("="*60)
        
        return True


def main():
    """
    Main execution function
    """
    # Configuration
    MODEL_NAME = "Random_Forest"  # Change to XGBoost, Gradient_Boosting, etc.
    DATA_SOURCE = "hopsworks"     # or "local"
    OUTPUT_DIR = "shap_analysis"
    
    # Initialize analyzer
    analyzer = AQIShapAnalyzer(
        model_name=MODEL_NAME,
        data_source=DATA_SOURCE
    )
    
    # Run full analysis
    analyzer.run_full_analysis(output_dir=OUTPUT_DIR)
    
    # Print top features
    importance_df = analyzer.get_feature_importance()
    if importance_df is not None:
        print("\n🏆 Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()