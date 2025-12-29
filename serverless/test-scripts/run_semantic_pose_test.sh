#!/bin/bash
# Run Semantic Segmentation & Hand Pose Integration Test
# =====================================================
#
# This script runs semantic segmentation (SAM/Detectron2) and hand pose estimation (MediaPipe)
# on test images and validates their intersection.

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Semantic Segmentation & Hand Pose Integration Test${NC}"
echo "=================================================================="

# Check MediaPipe service
echo -e "\n${BLUE}Checking MediaPipe service...${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MediaPipe service is running${NC}"
else
    echo -e "${YELLOW}⚠ MediaPipe service not running on port 8000${NC}"
    echo "  Start it with: cd serverless/mediapipe-service && ./start.sh"
    exit 1
fi

# Check SAM service
echo -e "\n${BLUE}Checking SAM service...${NC}"
SAM_URL=""
for port in 32800 32768 32770; do
    if curl -s http://localhost:${port}/health > /dev/null 2>&1; then
        SAM_URL="http://localhost:${port}"
        echo -e "${GREEN}✓ SAM service found on port ${port}${NC}"
        break
    fi
done

if [ -z "$SAM_URL" ]; then
    echo -e "${YELLOW}⚠ SAM service not found${NC}"
    echo "  Deploy with: cd serverless && ./deploy_egocentric_models.sh --sam"
fi

# Check Detectron2 service
echo -e "\n${BLUE}Checking Detectron2 service...${NC}"
DETECTRON_URL=""
for port in 32769 32770; do
    if curl -s http://localhost:${port}/health > /dev/null 2>&1; then
        DETECTRON_URL="http://localhost:${port}"
        echo -e "${GREEN}✓ Detectron2 service found on port ${port}${NC}"
        break
    fi
done

if [ -z "$DETECTRON_URL" ]; then
    echo -e "${YELLOW}⚠ Detectron2 service not found${NC}"
    echo "  Deploy with: cd serverless && ./deploy_egocentric_models.sh --detectron2"
fi

# Determine which semantic model to use
SEMANTIC_MODEL="sam"
SEMANTIC_URL="$SAM_URL"

if [ -z "$SAM_URL" ] && [ -n "$DETECTRON_URL" ]; then
    SEMANTIC_MODEL="detectron2"
    SEMANTIC_URL="$DETECTRON_URL"
    echo -e "\n${YELLOW}Using Detectron2 for semantic segmentation${NC}"
elif [ -z "$SAM_URL" ] && [ -z "$DETECTRON_URL" ]; then
    echo -e "\n${YELLOW}⚠ No semantic segmentation service available${NC}"
    echo "  Will test MediaPipe hand pose only"
    SEMANTIC_MODEL="none"
fi

# Run test
echo -e "\n${BLUE}Running integration test...${NC}"
cd "$(dirname "$0")"

if [ "$SEMANTIC_MODEL" = "none" ]; then
    echo "Testing MediaPipe hand pose only..."
    python test_mediapipe_ground_truth.py \
        --dataset-path test-data/egocentric-hands \
        --mediapipe-url http://localhost:8000 \
        --max-samples 10 \
        --output-dir test-results
else
    python test_semantic_and_pose_integration.py \
        --dataset-path test-data/egocentric-hands \
        --semantic-model "$SEMANTIC_MODEL" \
        --sam-url "$SAM_URL" \
        --detectron-url "$DETECTRON_URL" \
        --mediapipe-url http://localhost:8000 \
        --max-samples 10 \
        --output-dir test-results
fi

echo -e "\n${GREEN}✓ Test complete!${NC}"
echo "  Results saved to: test-results/"

