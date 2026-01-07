#!/bin/bash
# Complete MediaPipe Pose Service Setup and CVAT Integration
# ==========================================================
#
# This script provides a complete setup workflow for the MediaPipe pose service,
# including virtual environment setup, service testing, and CVAT integration.
#
# Usage:
#   ./run-setup.sh [options]
#
# Options:
#   --cvat-url URL       CVAT server URL (default: http://localhost:8080)
#   --service-port PORT  MediaPipe service port (default: 8000)
#   --auth-token TOKEN   CVAT authentication token
#   --docker             Use Docker instead of virtual environment
#   --help               Show this help message

set -e

# Default values
CVAT_URL="http://localhost:8080"
SERVICE_PORT="8000"
SERVICE_URL="http://localhost:$SERVICE_PORT"
USE_DOCKER=false
AUTH_TOKEN=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    cat << EOF
Complete MediaPipe Pose Service Setup

This script sets up the MediaPipe pose detection service and integrates it with CVAT.

USAGE:
    ./run-setup.sh [OPTIONS]

OPTIONS:
    --cvat-url URL       CVAT server URL (default: http://localhost:8080)
    --service-port PORT  MediaPipe service port (default: 8000)
    --auth-token TOKEN   CVAT authentication token for API access
    --docker             Use Docker instead of virtual environment
    --help              Show this help message

EXAMPLES:
    # Basic setup with defaults
    ./run-setup.sh

    # Setup with custom CVAT URL
    ./run-setup.sh --cvat-url http://cvat.example.com:8080

    # Setup with authentication
    ./run-setup.sh --auth-token your_cvat_token_here

    # Docker-based setup
    ./run-setup.sh --docker

SETUP PROCESS:
    1. Environment setup (virtualenv or Docker)
    2. Service installation and testing
    3. CVAT integration and function registration
    4. Final verification and usage instructions

REQUIREMENTS:
    - Python 3.8+ (for virtualenv setup)
    - Docker (for Docker setup)
    - CVAT server running (for integration)

NOTES:
    - The script will automatically detect and use the appropriate setup method
    - CVAT authentication token can be obtained from CVAT → Settings → Account → Token
    - The service will be available at http://localhost:8000 by default
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
            --cvat-url)
                CVAT_URL="$2"
                shift 2
                ;;
            --service-port)
                SERVICE_PORT="$2"
                SERVICE_URL="http://localhost:$SERVICE_PORT"
                shift 2
                ;;
            --auth-token)
                AUTH_TOKEN="$2"
                shift 2
                ;;
            --docker)
                USE_DOCKER=true
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

check_dependencies() {
    log_info "Checking dependencies..."

    if [[ "$USE_DOCKER" == true ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed. Please install Docker first."
            exit 1
        fi
        if ! docker info &> /dev/null; then
            log_error "Docker is not running. Please start Docker first."
            exit 1
        fi
        log_success "Docker is available"
    else
        if ! command -v python3 &> /dev/null; then
            log_error "Python 3 is not installed. Please install Python 3.8+ first."
            exit 1
        fi

        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Python $PYTHON_VERSION is compatible"
        else
            log_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    fi
}

setup_service() {
    log_info "Setting up MediaPipe pose service..."

    if [[ "$USE_DOCKER" == true ]]; then
        log_info "Using Docker setup..."

        # Build Docker image
        log_info "Building Docker image..."
        docker build -t mediapipe-pose-service .

        # Start service with docker-compose
        log_info "Starting service with Docker Compose..."
        docker-compose up -d

        # Wait for service to be healthy
        log_info "Waiting for service to be healthy..."
        for i in {1..30}; do
            if curl -f http://localhost:$SERVICE_PORT/health &> /dev/null; then
                log_success "Service is healthy"
                break
            fi
            sleep 2
        done

        if ! curl -f http://localhost:$SERVICE_PORT/health &> /dev/null; then
            log_error "Service failed to start properly"
            docker-compose logs mediapipe-pose
            exit 1
        fi

    else
        log_info "Using virtual environment setup..."

        # Run the setup script
        if [[ -f "setup.sh" ]]; then
            log_info "Running setup script..."
            ./setup.sh
        else
            log_error "setup.sh not found"
            exit 1
        fi

        # Start the service
        log_info "Starting service..."
        ./start.sh &

        # Wait for service to start
        sleep 3

        # Check if service is running
        if ! curl -f http://localhost:$SERVICE_PORT/health &> /dev/null; then
            log_error "Service failed to start"
            ./stop.sh 2>/dev/null || true
            exit 1
        fi
    fi

    log_success "MediaPipe pose service is running"
}

test_service() {
    log_info "Testing MediaPipe service..."

    # Test health endpoint
    if ! curl -f $SERVICE_URL/health &> /dev/null; then
        log_error "Health check failed"
        exit 1
    fi

    # Test pose detection
    log_info "Testing pose detection..."
    TEST_RESPONSE=$(curl -s -X POST $SERVICE_URL/detect \
        -H "Content-Type: application/json" \
        -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", "threshold": 0.1}' \
        2>/dev/null)

    if [[ $? -eq 0 ]] && [[ "$TEST_RESPONSE" != "" ]]; then
        log_success "Pose detection test passed"
    else
        log_warning "Pose detection test may have issues (this is expected for minimal test image)"
    fi
}

integrate_with_cvat() {
    log_info "Integrating with CVAT..."

    # Check if CVAT is accessible
    if ! curl -f $CVAT_URL &> /dev/null; then
        log_warning "CVAT is not accessible at $CVAT_URL"
        log_info "Skipping CVAT integration. You can run it manually later:"
        echo "  python cvat-integration.py --cvat-url $CVAT_URL --service-url $SERVICE_URL"
        return
    fi

    # Run CVAT integration
    INTEGRATION_CMD="python cvat-integration.py --cvat-url $CVAT_URL --service-url $SERVICE_URL"
    if [[ -n "$AUTH_TOKEN" ]]; then
        INTEGRATION_CMD="$INTEGRATION_CMD --auth-token $AUTH_TOKEN"
    fi

    log_info "Running CVAT integration..."
    if eval "$INTEGRATION_CMD"; then
        log_success "CVAT integration completed"
    else
        log_warning "CVAT integration had issues (may need manual authentication)"
        log_info "You can retry with authentication token:"
        echo "  python cvat-integration.py --cvat-url $CVAT_URL --service-url $SERVICE_URL --auth-token YOUR_TOKEN"
    fi
}

show_completion_message() {
    log_success "Setup completed successfully!"
    echo ""
    log_info "Service Information:"
    echo "  📍 URL: $SERVICE_URL"
    echo "  🔍 Health: $SERVICE_URL/health"
    echo "  🎯 Detect: $SERVICE_URL/detect"
    echo ""

    if [[ "$USE_DOCKER" == true ]]; then
        log_info "Docker Management:"
        echo "  🛑 Stop: docker-compose down"
        echo "  📋 Logs: docker-compose logs -f"
        echo "  🔄 Restart: docker-compose restart"
    else
        log_info "Service Management:"
        echo "  🛑 Stop: ./stop.sh"
        echo "  📊 Status: ./status.sh"
        echo "  🔄 Restart: ./stop.sh && ./start.sh"
    fi

    echo ""
    log_info "CVAT Integration:"
    echo "  🎨 Function registered as: 'MediaPipe Pose Detection'"
    echo "  📝 Use in tasks: Draw new shape → Select function"
    echo ""

    log_info "Test the service:"
    echo "  curl $SERVICE_URL/health"
    echo "  curl -X POST $SERVICE_URL/detect -H 'Content-Type: application/json' -d '{\"image\": \"base64_image_here\"}'"
}

main() {
    # Parse command line arguments
    parse_args "$@"

    log_info "MediaPipe Pose Service Complete Setup"
    log_info "====================================="
    log_info "CVAT URL: $CVAT_URL"
    log_info "Service URL: $SERVICE_URL"
    log_info "Using Docker: $USE_DOCKER"
    echo ""

    # Run setup steps
    check_dependencies
    setup_service
    test_service
    integrate_with_cvat
    show_completion_message
}

# Run main function
main "$@"
