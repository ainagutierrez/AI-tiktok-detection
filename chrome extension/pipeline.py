"""
python pipeline.py "path"  --methods [cnn, pretrained]
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime
import pandas as pd

from predict import CNNDetector
from pretrained_detector import PretrainedDetector
import config

import sys
print("Python executable:", sys.executable)


os.makedirs(config.LOGS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Master pipeline for running multiple detection methods
    """
    
    def __init__(self, methods=None, output_dir=None):
        """
        Initialize detection pipeline
        
        Args:
            methods: List of method names to use, or None for all enabled
            output_dir: Directory for saving results
        """
        self.output_dir = output_dir or config.RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        # Initialize detectors
        self.detectors = {}
        
        if methods is None:
            methods = [k for k, v in config.ENABLED_METHODS.items() if v]
        
        logger.info(f"Initializing detectors: {methods}")
        
        if 'cnn' in methods and config.ENABLED_METHODS['cnn']:
            try:
                logger.info("Initializing CNN detector...")
                self.detectors['cnn'] = CNNDetector(
                    model_path=config.CNN_MODEL_PATH,
                    sr=config.CNN_CONFIG['sr'],
                    duration=config.CNN_CONFIG['duration'],
                    n_mels=config.CNN_CONFIG['n_mels']
                )
                logger.info("CNN detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize CNN detector: {e}")
        
        if any(m.startswith('pretrained') for m in methods) and config.ENABLED_METHODS['pretrained']:
            for method in methods:
                if method == 'pretrained':
                    # Run all models
                    for model_name in config.PRETRAINED_MODELS:
                        detector_key = f"pretrained-{model_name.split('/')[-1]}"
                        try:
                            self.detectors[detector_key] = PretrainedDetector(model_name=model_name)
                            logger.info(f"Pretrained detector initialized: {model_name}")
                        except Exception as e:
                            logger.error(f"Failed to initialize pretrained detector {model_name}: {e}")
                elif method.startswith('pretrained:'):
                    # Run only the specific model
                    model_name = method.split(':', 1)[1]
                    detector_key = f"pretrained-{model_name.split('/')[-1]}"
                    try:
                        self.detectors[detector_key] = PretrainedDetector(model_name=model_name)
                        logger.info(f"Pretrained detector initialized: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize pretrained detector {model_name}: {e}")
        
        # FIXED: Moved this check outside the loop
        if not self.detectors:
            raise ValueError("No detectors were successfully initialized")
            
    def validate_audio_file(self, file_path: str) -> bool:
        """Check if file exists and has supported format"""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return False
        if path.suffix.lower() not in config.SUPPORTED_AUDIO_FORMATS:
            logger.warning(f"Unsupported format: {path.suffix}")
            return False
        return True
    
    def gather_audio_files(self, paths: list) -> list:
        """Expand list of files/folders into all valid audio files."""
        audio_files = []
        for path in paths:
            p = Path(path)
            if p.is_file() and p.suffix.lower() in config.SUPPORTED_AUDIO_FORMATS:
                audio_files.append(str(p))
            elif p.is_dir():
                # Add all supported audio files in this directory (non-recursive)
                files_in_dir = [str(f) for f in p.iterdir() if f.suffix.lower() in config.SUPPORTED_AUDIO_FORMATS]
                audio_files.extend(files_in_dir)
            else:
                logger.warning(f"Skipping unsupported path: {path}")
        return audio_files

    
    def run_detection(self, audio_files: Union[str, List[str]]) -> Dict:
        """
        Run all detection methods on provided audio files
        
        Args:
            audio_files: Single file path or list of file paths
            
        Returns:
            Dictionary with detection results
        """
        if isinstance(audio_files, str):
            audio_files = [audio_files]
        
        # Validate files
        audio_files = [f for f in audio_files if self.validate_audio_file(f)]
        
        if not audio_files:
            raise ValueError("No valid audio files provided")
        
        logger.info(f"Processing {len(audio_files)} audio files")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'methods': list(self.detectors.keys()),
                'cnn_model': config.CNN_MODEL_PATH if 'cnn' in self.detectors else None,
                'pretrained_model': config.DEFAULT_PRETRAINED_MODEL if 'pretrained' in self.detectors else None
            },
            'files': []
        }
        
        for audio_file in audio_files:
            logger.info(f"Processing: {audio_file}")
            file_result = {
                'file': audio_file,
                'filename': os.path.basename(audio_file),
                'predictions': {},
                'ground_truth': config.GROUND_TRUTH.get(os.path.basename(audio_file))
            }
            
            # Run each detector
            for method_name, detector in self.detectors.items():
                logger.info(f"  Running {method_name} detector")
                prediction = detector.predict(audio_file)
                file_result['predictions'][method_name] = prediction
                
                if prediction['success']:
                    logger.info(f"    Result: {prediction['label']} (prob: {prediction['probability']:.4f})")
                else:
                    logger.error(f"    Error: {prediction['error']}")
            
            results['files'].append(file_result)
        
        return results
    
    def evaluate_results(self, results: Dict) -> Dict:
        """
        Calculate performance metrics if ground truth is available
        
        Args:
            results: Results dictionary from run_detection
            
        Returns:
            Dictionary with evaluation metrics
        """
        evaluation = {
            'per_method': {},
            'summary': {}
        }
        
        # Check if we have any ground truth
        has_ground_truth = any(f['ground_truth'] is not None for f in results['files'])
        
        if not has_ground_truth:
            logger.warning("No ground truth available for evaluation")
            return evaluation
        
        # Calculate metrics per method
        for method_name in self.detectors.keys():
            correct = 0
            total = 0
            errors = 0
            
            # ADDED: Debug logging
            logger.debug(f"\nEvaluating method: {method_name}")
            
            for file_result in results['files']:
                if file_result['ground_truth'] is None:
                    continue
                
                prediction = file_result['predictions'][method_name]
                if not prediction['success']:
                    errors += 1
                    logger.debug(f"  {file_result['filename']}: ERROR - {prediction['error']}")
                    continue
                
                total += 1
                # Normalize labels for comparison
                pred_label = prediction['label'].upper()
                true_label = file_result['ground_truth'].upper()
                
                # ADDED: Debug logging
                logger.debug(f"  {file_result['filename']}: pred={pred_label}, true={true_label}, match={pred_label == true_label}")
                
                if pred_label == true_label:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            evaluation['per_method'][method_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'errors': errors
            }
            
            logger.info(f"Method {method_name}: Accuracy = {accuracy:.2%} ({correct}/{total})")
        
        return evaluation
    
    def save_results(self, results: Dict, evaluation: Dict, format='json'):
        """
        Save results to file
        
        Args:
            results: Detection results
            evaluation: Evaluation metrics
            format: Output format ('json', 'csv', 'table')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            output_file = os.path.join(self.output_dir, f'results_{timestamp}.json')
            with open(output_file, 'w') as f:
                json.dump({'results': results, 'evaluation': evaluation}, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        elif format == 'csv':
            # Flatten results for CSV
            rows = []
            for file_result in results['files']:
                row = {
                    'file': file_result['filename'],
                    'ground_truth': file_result['ground_truth']
                }
                for method_name, prediction in file_result['predictions'].items():
                    row[f'{method_name}_label'] = prediction['label']
                    row[f'{method_name}_probability'] = prediction['probability']
                    row[f'{method_name}_success'] = prediction['success']
                rows.append(row)
            
            df = pd.DataFrame(rows)
            output_file = os.path.join(self.output_dir, f'results_{timestamp}.csv')
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        elif format == 'table':
            # Pretty print table
            output_file = os.path.join(self.output_dir, f'results_{timestamp}.txt')
            with open(output_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("DETECTION RESULTS\n")
                f.write("="*80 + "\n\n")
                
                for file_result in results['files']:
                    f.write(f"File: {file_result['filename']}\n")
                    f.write(f"Ground Truth: {file_result['ground_truth']}\n")
                    f.write("-"*80 + "\n")
                    
                    for method_name, prediction in file_result['predictions'].items():
                        if prediction['success']:
                            f.write(f"  {method_name:15s}: {prediction['label']:6s} "
                                  f"(prob: {prediction['probability']:.4f})\n")
                        else:
                            f.write(f"  {method_name:15s}: ERROR - {prediction['error']}\n")
                    f.write("\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("EVALUATION SUMMARY\n")
                f.write("="*80 + "\n")
                
                for method_name, metrics in evaluation.get('per_method', {}).items():
                    f.write(f"\n{method_name}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.2%}\n")
                    f.write(f"  Correct: {metrics['correct']}/{metrics['total']}\n")
                    f.write(f"  Errors: {metrics['errors']}\n")
            
            logger.info(f"Results saved to {output_file}")
    
    def run(self, audio_files: Union[str, List[str]], output_format=None):
        """
        Complete pipeline execution
        
        Args:
            audio_files: Audio file(s) to process
            output_format: Output format override
        """
        logger.info("Starting detection pipeline")
        
        # Run detection
        results = self.run_detection(audio_files)
        
        # Evaluate
        evaluation = self.evaluate_results(results)
        
        # Save results
        format = output_format or config.OUTPUT_FORMAT
        self.save_results(results, evaluation, format=format)
        
        logger.info("Pipeline execution completed")
        
        return results, evaluation


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run audio detection pipeline')
    parser.add_argument('files', nargs='+', help='Audio files to process')
    parser.add_argument('--methods', nargs='+', choices=['cnn', 'pretrained'], 
                       help='Detection methods to use')
    parser.add_argument('--output-format', choices=['json', 'csv', 'table'],
                       default=config.OUTPUT_FORMAT, help='Output format')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = DetectionPipeline(
        methods=args.methods,
        output_dir=args.output_dir
    )
    
        # Gather all audio files from files/folders
    audio_files = pipeline.gather_audio_files(args.files)
    if not audio_files:
        logger.error("No valid audio files found.")
        return

    # Run pipeline
    results, evaluation = pipeline.run(
        audio_files=audio_files,
        output_format=args.output_format
    )

    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Files processed: {len(results['files'])}")
    print(f"Methods used: {', '.join(results['config']['methods'])}")
    
    if evaluation.get('per_method'):
        print("\nAccuracy per method:")
        for method, metrics in evaluation['per_method'].items():
            print(f"  {method}: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    main()