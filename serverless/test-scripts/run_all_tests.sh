#!/bin/bash
# Complete Egocentric Model Testing Suite Runner
# ===============================================
#
# Runs the complete testing suite for SAM, SAM Auto, Detectron2, and MediaPipe models
# on egocentric vision datasets. Includes automated setup, testing, and analysis.
#
# Usage:
#   ./run_all_tests.sh [options]
#
# Options:
#   --dataset-dir DIR        Dataset directory (default: ./test-data)
#   --results-dir DIR        Results directory (default: ./test-results)
#   --models MODEL1,MODEL2   Models to test (default: all)
#   --difficulty LEVEL       Difficulty level (default: all)
#   --max-samples N          Max samples per test (default: 50)
#   --skip-dataset           Skip dataset download
#   --generate-report        Generate analysis report
#   --help                   Show this help

set -e

# Default values
DATASET_DIR="./test-data"
RESULTS_DIR="./test-results"
MODELS="sam,sam-auto,detectron2,mediapipe"
DIFFICULTY="all"
MAX_SAMPLES=50
SKIP_DATASET=false
GENERATE_REPORT=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    cat << EOF
Complete Egocentric Model Testing Suite

Runs comprehensive testing of SAM, SAM Auto, Detectron2, and MediaPipe models on egocentric datasets.

USAGE:
    ./run_all_tests.sh [OPTIONS]

OPTIONS:
    --dataset-dir DIR        Dataset directory (default: ./test-data)
    --results-dir DIR        Results directory (default: ./test-results)
    --models MODEL1,MODEL2   Models to test: sam, detectron2, mediapipe (default: all)
    --difficulty LEVEL       Difficulty: easy, medium, hard, all (default: all)
    --max-samples N          Max samples per test (default: 50)
    --skip-dataset           Skip dataset download/setup
    --generate-report        Generate analysis report (default: true)
    --help                   Show this help

EXAMPLES:
    # Run all tests with defaults
    ./run_all_tests.sh

    # Test only SAM and MediaPipe on easy images
    ./run_all_tests.sh --models sam,mediapipe --difficulty easy --max-samples 25

    # Skip dataset setup, just run tests
    ./run_all_tests.sh --skip-dataset

REQUIREMENTS:
    - Python 3.8+ with required packages
    - Model services running (SAM, Detectron2, MediaPipe)
    - Sufficient disk space for datasets and results

OUTPUT:
    - Test results in ./test-results/
    - Analysis reports in ./test-results/analysis/
EOF
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset-dir)
                DATASET_DIR="$2"
                shift 2
                ;;
            --results-dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            --models)
                MODELS="$2"
                shift 2
                ;;
            --difficulty)
                DIFFICULTY="$2"
                shift 2
                ;;
            --max-samples)
                MAX_SAMPLES="$2"
                shift 2
                ;;
            --skip-dataset)
                SKIP_DATASET=true
                shift
                ;;
            --generate-report)
                GENERATE_REPORT=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

check_requirements() {
    log_info "Checking requirements..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not available"
        exit 1
    fi

    # Check if required packages are installed
    python3 -c "
import sys
required_packages = ['requests', 'PIL', 'tqdm']
missing = []

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        missing.append(package)

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
" || {
        log_error "Required Python packages are missing. Install with:"
        echo "  pip install requests Pillow tqdm matplotlib scipy"
        exit 1
    }

    log_success "Requirements check passed"
}

setup_dataset() {
    if [[ "$SKIP_DATASET" == true ]]; then
        log_info "Skipping dataset setup"
        return
    fi

    log_info "Setting up test dataset..."

    if [[ ! -f "download_real_images.py" ]]; then
        log_error "Real image download script not found"
        exit 1
    fi

    # Run real image download
    python3 download_real_images.py \
        --category all \
        --count "$MAX_SAMPLES"

    log_success "Dataset setup completed"
}

check_services() {
    log_info "Checking model services..."

    # Check SAM service
    if [[ "$MODELS" == *"sam"* ]]; then
        if curl -f http://localhost:32768/health &> /dev/null; then
            log_success "SAM service is running"
        else
            log_warning "SAM service not detected at http://localhost:32768"
            echo "  Start with: nuctl get functions | grep sam"
        fi
    fi

    # Check Detectron2 service
    if [[ "$MODELS" == *"detectron2"* ]]; then
        if curl -f http://localhost:32769/health &> /dev/null; then
            log_success "Detectron2 service is running"
        else
            log_warning "Detectron2 service not detected at http://localhost:32769"
            echo "  Start with: nuctl get functions | grep detectron"
        fi
    fi

    # Check MediaPipe service
    if [[ "$MODELS" == *"mediapipe"* ]]; then
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "MediaPipe service is running"
        else
            log_warning "MediaPipe service not detected at http://localhost:8000"
            echo "  Start with: cd ../mediapipe-service && ./start.sh"
        fi
    fi
}

run_tests() {
    log_info "Running model tests..."

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    # Split models
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

    for model in "${MODEL_ARRAY[@]}"; do
        case $model in
            sam)
                log_info "Testing SAM model..."
                python3 test_sam_egocentric.py \
                    --dataset "$DATASET_DIR" \
                    --difficulty "$DIFFICULTY" \
                    --max-samples "$MAX_SAMPLES" \
                    --output-dir "$RESULTS_DIR" \
                    --sam-url "http://localhost:32768"
                ;;
            sam-auto)
                log_info "Testing SAM Auto model..."
                python3 test_sam_auto_egocentric.py \
                    --dataset "$DATASET_DIR" \
                    --difficulty "$DIFFICULTY" \
                    --max-samples "$MAX_SAMPLES" \
                    --output-dir "$RESULTS_DIR" \
                    --sam-auto-url "http://localhost:32770"
                ;;
            detectron2)
                log_info "Testing Detectron2 model..."
                python3 test_detectron2_egocentric.py \
                    --dataset "$DATASET_DIR" \
                    --difficulty "$DIFFICULTY" \
                    --max-samples "$MAX_SAMPLES" \
                    --output-dir "$RESULTS_DIR" \
                    --detectron-url "http://localhost:32769"
                ;;
            mediapipe)
                log_info "Testing MediaPipe model..."
                python3 test_mediapipe_egocentric.py \
                    --dataset "$DATASET_DIR" \
                    --difficulty "$DIFFICULTY" \
                    --max-samples "$MAX_SAMPLES" \
                    --output-dir "$RESULTS_DIR" \
                    --mediapipe-url "http://localhost:8000"
                ;;
            *)
                log_warning "Unknown model: $model (skipping)"
                ;;
        esac
    done

    log_success "Model tests completed"
}

generate_analysis() {
    if [[ "$GENERATE_REPORT" != true ]]; then
        return
    fi

    log_info "Generating analysis report..."

    ANALYSIS_DIR="$RESULTS_DIR/analysis"
    mkdir -p "$ANALYSIS_DIR"

    # Run analysis script
    python3 analyze_results.py \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$ANALYSIS_DIR" \
        --compare-models \
        --generate-report \
        --create-plots

    log_success "Analysis report generated in $ANALYSIS_DIR"
}

show_summary() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                 🧪 TESTING COMPLETE 🧪                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Configuration:"
    echo "   📁 Dataset: $DATASET_DIR"
    echo "   📄 Results: $RESULTS_DIR"
    echo "   🤖 Models: $MODELS"
    echo "   🎯 Difficulty: $DIFFICULTY"
    echo "   📏 Max Samples: $MAX_SAMPLES"
    echo ""

    if [[ "$GENERATE_REPORT" == true ]]; then
        ANALYSIS_DIR="$RESULTS_DIR/analysis"
        echo "📈 Analysis Reports:"
        echo "   📄 JSON Report: $ANALYSIS_DIR/comprehensive_report.json"
        echo "   📄 Markdown Report: $ANALYSIS_DIR/analysis_report.md"
        echo "   📊 Charts: $ANALYSIS_DIR/*.png"
        echo ""
    fi

    echo "🎯 Next Steps:"
    echo "   1. Review test results in $RESULTS_DIR"
    echo "   2. Check analysis reports for insights"
    echo "   3. Compare model performance for your use case"
    echo "   4. Fine-tune model selection based on requirements"
    echo ""

    echo "💡 Quick Commands:"
    echo "   # View results summary"
    echo "   ls -la $RESULTS_DIR/"
    echo ""
    echo "   # Read analysis report"
    echo "   cat $RESULTS_DIR/analysis/analysis_report.md"
    echo ""
}

main() {
    log_info "Egocentric Model Testing Suite Runner"
    log_info "====================================="
    log_info "Dataset: $DATASET_DIR"
    log_info "Results: $RESULTS_DIR"
    log_info "Models: $MODELS"
    log_info "Difficulty: $DIFFICULTY"
    echo ""

    # Parse command line arguments
    parse_args "$@"

    # Run testing pipeline
    check_requirements
    setup_dataset
    check_services
    run_tests
    generate_analysis
    show_summary

    log_success "All testing completed successfully! 🎉"
}

# Run main function
main "$@"
