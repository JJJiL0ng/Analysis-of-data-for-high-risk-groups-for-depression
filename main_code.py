"""
Mental Health Prediction - Main Controller
==========================================

This main controller imports and executes the existing analysis modules:
1. encoding.py - Data encoding
2. scaling.py - Data scaling  
3. evaluation_kfold.py - Classification models and evaluation
4. K-means.py - Clustering analysis

Simply run this file to execute the complete pipeline using existing modules.

Usage:
    python main_code.py

Requirements:
    - anxiety_depression_data.csv must be in the root directory
    - All module files must be in the same directory
    - All required packages must be installed
"""

import os
import sys
import subprocess
import importlib.util
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class MentalHealthPipelineController:
    """Controller class that manages the execution of existing analysis modules"""
    
    def __init__(self):
        """Initialize the pipeline controller"""
        self.required_files = [
            "anxiety_depression_data.csv",
            "encoding.py", 
            "scaling.py",
            "evaluation_kfold.py",
            "K-means.py"
        ]
        
        self.results_dir = "results"
        self.create_results_directory()
        
        print("="*60)
        print("MENTAL HEALTH PREDICTION PIPELINE CONTROLLER")
        print("="*60)
        print(f"Initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def create_results_directory(self):
        """Create results directory structure"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.results_dir}/models", exist_ok=True)
        print(f"✓ Results directory created: {self.results_dir}/")
    
    def check_requirements(self):
        """Check if all required files exist"""
        print("\n" + "="*50)
        print("CHECKING REQUIREMENTS")
        print("="*50)
        
        missing_files = []
        for file in self.required_files:
            if os.path.exists(file):
                print(f"✓ {file}")
            else:
                print(f"❌ {file} - NOT FOUND")
                missing_files.append(file)
        
        if missing_files:
            print(f"\n❌ Missing files: {missing_files}")
            print("Please ensure all required files are in the root directory.")
            return False
        
        print("\n✅ All required files found!")
        return True
    
    def check_packages(self):
        """Check if required packages are installed"""
        print("\n" + "="*50)
        print("CHECKING PACKAGES")
        print("="*50)
        
        # Package name mapping: display_name -> import_name
        required_packages = {
            'numpy': 'numpy',
            'pandas': 'pandas', 
            'scikit-learn': 'sklearn',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'xgboost': 'xgboost',
            'imbalanced-learn': 'imblearn'
        }
        
        missing_packages = []
        for display_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"✓ {display_name}")
            except ImportError:
                print(f"❌ {display_name} - NOT INSTALLED")
                missing_packages.append(display_name)
        
        if missing_packages:
            print(f"\n❌ Missing packages: {missing_packages}")
            print("Please install missing packages:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("\n✅ All required packages found!")
        return True
    
    def run_module(self, module_name, description):
        """Execute a Python module and handle errors"""
        print(f"\n" + "="*50)
        print(f"EXECUTING: {description}")
        print(f"Module: {module_name}")
        print("="*50)
        
        try:
            # Method 1: Using subprocess (more isolated)
            result = subprocess.run([sys.executable, module_name], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully!")
                if result.stdout:
                    print("Output preview:")
                    # Show last few lines of output
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:  # Show last 5 lines
                        if line.strip():  # Skip empty lines
                            print(f"  {line}")
                return True
            else:
                print(f"❌ {description} failed!")
                print("Error:", result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out (>5 minutes)")
            return False
        except Exception as e:
            print(f"❌ Error executing {module_name}: {str(e)}")
            return False
    
    def run_module_direct_import(self, module_name, description):
        """Alternative method: Direct import and execution"""
        print(f"\n" + "="*50)
        print(f"EXECUTING: {description}")
        print(f"Module: {module_name}")
        print("="*50)
        
        try:
            # Load module specification
            spec = importlib.util.spec_from_file_location("temp_module", module_name)
            if spec is None:
                print(f"❌ Could not load module specification for {module_name}")
                return False
            
            # Create module from specification
            module = importlib.util.module_from_spec(spec)
            
            # Execute the module
            spec.loader.exec_module(module)
            
            print(f"✅ {description} completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error executing {module_name}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            # Don't print full traceback for cleaner output
            return False
    
    def organize_results(self):
        """Organize and move results to results directory"""
        print("\n" + "="*50)
        print("ORGANIZING RESULTS")
        print("="*50)
        
        # Files that might be generated by the modules
        potential_outputs = [
            "Robustscaling_Q1.csv",
            "kmeans_clustering_results.csv",
            "encoded_data.csv"
        ]
        
        moved_files = []
        for file in potential_outputs:
            if os.path.exists(file):
                try:
                    import shutil
                    destination = os.path.join(self.results_dir, file)
                    shutil.move(file, destination)
                    moved_files.append(file)
                    print(f"✓ Moved {file} to {self.results_dir}/")
                except Exception as e:
                    print(f"⚠️ Could not move {file}: {str(e)}")
        
        if moved_files:
            print(f"\n✅ Organized {len(moved_files)} result files")
        else:
            print("\n📝 No additional files to organize")
        
        return moved_files
    
    def generate_execution_summary(self, results):
        """Generate a summary of the pipeline execution"""
        print("\n" + "="*50)
        print("GENERATING EXECUTION SUMMARY")
        print("="*50)
        
        summary_content = f"""
Mental Health Prediction Pipeline - Execution Summary
====================================================
Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Pipeline Modules Executed:
==========================

1. Data Encoding (encoding.py)
   Status: {'✅ SUCCESS' if results.get('encoding', False) else '❌ FAILED'}
   Description: Applied ordinal and one-hot encoding to categorical variables

2. Data Scaling (scaling.py)  
   Status: {'✅ SUCCESS' if results.get('scaling', False) else '❌ FAILED'}
   Description: Applied robust scaling using Q1 method to numerical features

3. Classification & Evaluation (evaluation_kfold.py)
   Status: {'✅ SUCCESS' if results.get('classification', False) else '❌ FAILED'}
   Description: Trained multiple models and performed k-fold cross-validation
   Models: Decision Tree, Random Forest, XGBoost, Bagging Classifier

4. Clustering Analysis (K-means.py)
   Status: {'✅ SUCCESS' if results.get('clustering', False) else '❌ FAILED'}
   Description: Performed K-means clustering analysis with optimal cluster detection

Overall Pipeline Status:
========================
Total Modules: 4
Successful: {sum(results.values())}
Failed: {4 - sum(results.values())}

Success Rate: {(sum(results.values()) / 4) * 100:.1f}%

Generated Files:
===============
- Check the results/ directory for all output files
- Visualization plots may be displayed during execution
- CSV files with processed data and results

Notes:
======
- Each module was executed independently
- Some modules may generate plots that display automatically
- Check individual module outputs for detailed results
- If any module failed, check the error messages above

Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Save summary to file
        summary_path = os.path.join(self.results_dir, "pipeline_execution_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        print(summary_content)
        print(f"✓ Summary saved to: {summary_path}")
        
        return summary_content
    
    def run_complete_pipeline(self):
        """Execute the complete analysis pipeline using existing modules"""
        print("🚀 Starting Mental Health Prediction Pipeline...")
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ Pipeline aborted due to missing files.")
            return False
        
        # Check packages
        if not self.check_packages():
            print("\n❌ Pipeline aborted due to missing packages.")
            return False
        
        # Define execution plan with CORRECT file names
        execution_plan = [
            ("encoding.py", "Data Encoding", "encoding"),
            ("scaling.py", "Data Scaling & Preprocessing", "scaling"), 
            ("evaluation_kfold.py", "Classification Models & Evaluation", "classification"),
            ("K-means.py", "K-means Clustering Analysis", "clustering")
        ]
        
        # Execute modules
        results = {}
        successful_modules = 0
        
        for module_file, description, key in execution_plan:
            print(f"\n⏳ Preparing to execute: {description}")
            
            # Try subprocess first, fallback to direct import
            success = self.run_module(module_file, description)
            
            if not success:
                print(f"🔄 Retrying with direct import method...")
                success = self.run_module_direct_import(module_file, description)
            
            results[key] = success
            if success:
                successful_modules += 1
            
            # Small delay between modules
            import time
            time.sleep(1)
        
        # Organize results
        self.organize_results()
        
        # Generate summary
        self.generate_execution_summary(results)
        
        # Final status
        print("\n" + "="*60)
        if successful_modules == len(execution_plan):
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("✅ All modules executed successfully")
        else:
            print("⚠️ PIPELINE COMPLETED WITH ISSUES")
            print("="*60)
            print(f"✅ {successful_modules}/{len(execution_plan)} modules completed successfully")
            print("❌ Some modules failed - check error messages above")
        
        print(f"\n📁 Results saved to: {self.results_dir}/")
        print("📊 Check for visualization plots that may have been displayed")
        print("📋 Full execution summary available in pipeline_execution_summary.txt")
        
        return successful_modules == len(execution_plan)

def print_usage_info():
    """Print usage information and requirements"""
    print("""
Usage Information
================

This controller executes the following modules in order:
1. encoding.py - Data encoding
2. scaling.py - Data scaling  
3. evaluation_kfold.py - Classification and evaluation
4. K-means.py - Clustering analysis

Required Files (must be in same directory):
- anxiety_depression_data.csv (dataset)
- encoding.py
- scaling.py
- evaluation_kfold.py
- K-means.py

The pipeline will:
- Execute each module sequentially
- Handle errors gracefully
- Organize results in results/ directory
- Generate execution summary
- Display progress and status updates

To run: python main_code.py
    """)

def main():
    """Main function to run the complete pipeline"""
    print_usage_info()
    
    # Initialize controller
    controller = MentalHealthPipelineController()
    
    # Run pipeline
    success = controller.run_complete_pipeline()
    
    if success:
        print("\n✨ All analysis completed successfully!")
        print("Check the results/ directory for all outputs.")
    else:
        print("\n⚠️ Pipeline completed with some issues.")
        print("Check the execution summary for details.")

if __name__ == "__main__":
    main()